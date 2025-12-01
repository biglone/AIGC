# Agent系统架构与实现

## 目录
1. [Agent基础](#agent基础)
2. [工具使用](#工具使用)
3. [规划与推理](#规划与推理)
4. [记忆管理](#记忆管理)
5. [多Agent协作](#多agent协作)

---

## Agent基础

### 什么是AI Agent？

**定义：** 能够感知环境、自主决策并采取行动以达成目标的智能体。

```
Agent = LLM + Memory + Planning + Tools

传统LLM：问题 → 回答
Agent：目标 → 观察 → 思考 → 行动 → 观察 → ... → 完成
```

### Agent架构

```
┌─────────────────────────────────────────┐
│          Perception (感知)               │
│  - 接收用户输入                           │
│  - 观察环境状态                           │
│  - 理解任务目标                           │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│          Planning (规划)                 │
│  - 分解复杂任务                           │
│  - 制定执行计划                           │
│  - 选择工具和策略                         │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│          Action (行动)                   │
│  - 调用工具/API                          │
│  - 执行代码                               │
│  - 生成内容                               │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│          Memory (记忆)                   │
│  - 短期记忆：对话历史                     │
│  - 长期记忆：知识库                       │
│  - 工作记忆：任务状态                     │
└─────────────────────────────────────────┘
```

### 基础Agent实现

```python
class BaseAgent:
    def __init__(self, llm, tools, memory):
        self.llm = llm
        self.tools = tools
        self.memory = memory

    def run(self, task):
        """
        Agent主循环
        """
        # 1. 感知：理解任务
        self.memory.add_task(task)

        # 2. 循环：思考-行动
        max_iterations = 10
        for i in range(max_iterations):
            # 思考：决定下一步行动
            thought = self.think()

            # 行动：执行决策
            action, action_input = self.decide_action(thought)

            if action == "FINISH":
                return action_input

            # 执行工具
            observation = self.execute(action, action_input)

            # 更新记忆
            self.memory.add_step(thought, action, observation)

        return "达到最大迭代次数，任务未完成"

    def think(self):
        """
        基于当前状态思考下一步
        """
        prompt = f"""
        任务：{self.memory.task}

        历史步骤：
        {self.memory.get_history()}

        可用工具：
        {self.get_tools_description()}

        下一步应该做什么？请思考。
        """

        return self.llm(prompt)

    def decide_action(self, thought):
        """
        从思考中提取行动
        """
        prompt = f"""
        思考：{thought}

        请决定下一步行动，格式：
        Action: [工具名称或FINISH]
        Action Input: [输入参数]
        """

        response = self.llm(prompt)
        action = parse_action(response)
        action_input = parse_action_input(response)

        return action, action_input

    def execute(self, action, action_input):
        """
        执行工具调用
        """
        if action not in self.tools:
            return f"错误：工具 {action} 不存在"

        try:
            result = self.tools[action](action_input)
            return result
        except Exception as e:
            return f"错误：{str(e)}"
```

---

## 工具使用

### Function Calling

**OpenAI Function Calling：**

```python
# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，如北京"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索网络信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# 调用
from openai import OpenAI
client = OpenAI()

messages = [
    {"role": "user", "content": "北京明天天气怎么样？"}
]

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# 处理工具调用
message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        # 执行函数
        if function_name == "get_weather":
            result = get_weather(**arguments)
        elif function_name == "search_web":
            result = search_web(**arguments)

        # 将结果返回给模型
        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

    # 继续对话
    final_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    )
```

### 工具注册与管理

```python
from typing import Callable, Dict, Any
from pydantic import BaseModel, Field

class Tool(BaseModel):
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable

class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, name: str, description: str):
        """装饰器：注册工具"""
        def decorator(func: Callable):
            # 从函数签名提取参数
            import inspect
            sig = inspect.signature(func)
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }

            for param_name, param in sig.parameters.items():
                param_info = {
                    "type": "string",
                    "description": param.annotation.__doc__ or ""
                }
                parameters["properties"][param_name] = param_info

                if param.default == inspect.Parameter.empty:
                    parameters["required"].append(param_name)

            tool = Tool(
                name=name,
                description=description,
                parameters=parameters,
                func=func
            )
            self.tools[name] = tool
            return func

        return decorator

    def get_tool_schemas(self):
        """获取OpenAI格式的工具定义"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools.values()
        ]

    def execute(self, name: str, arguments: Dict[str, Any]):
        """执行工具"""
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")

        return self.tools[name].func(**arguments)

# 使用示例
registry = ToolRegistry()

@registry.register(
    name="calculator",
    description="执行数学计算"
)
def calculator(expression: str):
    """计算数学表达式"""
    return eval(expression)

@registry.register(
    name="web_search",
    description="搜索网络"
)
def web_search(query: str, num_results: int = 5):
    """搜索网络信息"""
    # 实际实现
    import requests
    response = requests.get(
        "https://api.search.com/search",
        params={"q": query, "n": num_results}
    )
    return response.json()

# 获取工具定义
tools = registry.get_tool_schemas()

# 执行工具
result = registry.execute("calculator", {"expression": "2+2"})
```

### 代码解释器

```python
import sys
from io import StringIO

class CodeInterpreter:
    """安全的代码执行环境"""

    def __init__(self, timeout=5):
        self.timeout = timeout
        self.allowed_modules = {
            'math', 'statistics', 'datetime',
            'json', 're', 'collections'
        }

    def execute(self, code: str):
        """
        在受限环境中执行Python代码
        """
        # 捕获stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # 创建受限的globals
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    # ... 其他安全函数
                }
            }

            # 允许导入特定模块
            for module in self.allowed_modules:
                safe_globals[module] = __import__(module)

            # 执行代码（带超时）
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("代码执行超时")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)

            exec(code, safe_globals)

            signal.alarm(0)  # 取消超时

            # 获取输出
            output = sys.stdout.getvalue()

            return {
                "success": True,
                "output": output,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e)
            }

        finally:
            sys.stdout = old_stdout

# 使用示例
interpreter = CodeInterpreter()

result = interpreter.execute("""
import math

def calculate_circle_area(radius):
    return math.pi * radius ** 2

print(f"半径为5的圆的面积：{calculate_circle_area(5)}")
""")

print(result["output"])
# 输出：半径为5的圆的面积：78.53981633974483
```

---

## 规划与推理

### ReAct框架

**Reasoning + Acting：交替推理和行动**

```python
class ReActAgent:
    """ReAct Agent实现"""

    REACT_PROMPT = """
    你可以通过"思考-行动-观察"循环来解决问题。

    Thought: 思考下一步应该做什么
    Action: [工具名称]
    Action Input: [工具输入]
    Observation: [工具输出]
    ... (重复直到找到答案)
    Thought: 我现在知道答案了
    Final Answer: [最终答案]

    可用工具：
    {tools}

    问题：{question}

    开始！
    """

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, question):
        prompt = self.REACT_PROMPT.format(
            tools=self.format_tools(),
            question=question
        )

        history = []
        max_steps = 10

        for step in range(max_steps):
            # LLM生成下一步
            response = self.llm(prompt + "\n".join(history))

            # 解析响应
            if "Final Answer:" in response:
                answer = response.split("Final Answer:")[1].strip()
                return answer

            # 提取思考和行动
            thought = self.extract_thought(response)
            action = self.extract_action(response)
            action_input = self.extract_action_input(response)

            # 执行行动
            observation = self.execute_tool(action, action_input)

            # 更新历史
            history.append(f"Thought: {thought}")
            history.append(f"Action: {action}")
            history.append(f"Action Input: {action_input}")
            history.append(f"Observation: {observation}")

        return "达到最大步数限制"

# 示例
agent = ReActAgent(llm=gpt4, tools=tool_registry)

result = agent.run("2024年奥运会在哪里举办？参赛国家有多少个？")

# 执行过程：
# Thought: 我需要查找2024年奥运会的信息
# Action: web_search
# Action Input: "2024年奥运会举办地"
# Observation: 2024年奥运会在法国巴黎举办
#
# Thought: 现在我需要查找参赛国家数量
# Action: web_search
# Action Input: "2024巴黎奥运会参赛国家数量"
# Observation: 共有206个国家和地区参赛
#
# Thought: 我现在知道答案了
# Final Answer: 2024年奥运会在法国巴黎举办，共有206个国家和地区参赛。
```

### Plan-and-Execute

**先规划，再执行：**

```python
class PlanAndExecuteAgent:
    """
    计划-执行框架
    1. 将任务分解为子任务
    2. 逐步执行子任务
    3. 根据结果调整计划
    """

    PLANNER_PROMPT = """
    将以下任务分解为具体的子任务步骤：

    任务：{task}

    输出格式：
    1. [子任务1]
    2. [子任务2]
    ...
    """

    EXECUTOR_PROMPT = """
    执行以下子任务：

    子任务：{subtask}

    已完成步骤的结果：
    {previous_results}

    可用工具：
    {tools}

    请完成这个子任务。
    """

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def plan(self, task):
        """生成执行计划"""
        prompt = self.PLANNER_PROMPT.format(task=task)
        response = self.llm(prompt)

        # 解析子任务列表
        subtasks = self.parse_subtasks(response)
        return subtasks

    def execute_subtask(self, subtask, previous_results):
        """执行单个子任务"""
        prompt = self.EXECUTOR_PROMPT.format(
            subtask=subtask,
            previous_results=previous_results,
            tools=self.format_tools()
        )

        # 使用ReAct执行子任务
        react_agent = ReActAgent(self.llm, self.tools)
        result = react_agent.run(prompt)

        return result

    def run(self, task):
        # 1. 规划
        subtasks = self.plan(task)
        print(f"计划：\n{subtasks}")

        # 2. 执行
        results = []
        for i, subtask in enumerate(subtasks):
            print(f"\n执行步骤 {i+1}: {subtask}")

            result = self.execute_subtask(subtask, results)
            results.append({
                "subtask": subtask,
                "result": result
            })

            print(f"结果: {result}")

        # 3. 总结
        summary = self.summarize(task, results)
        return summary

# 示例
agent = PlanAndExecuteAgent(llm=gpt4, tools=tool_registry)

task = "分析比特币过去一周的价格走势，预测未来趋势"

result = agent.run(task)

# 计划：
# 1. 获取比特币过去一周的价格数据
# 2. 计算关键统计指标（涨跌幅、波动率等）
# 3. 分析价格趋势
# 4. 基于历史数据预测未来走势
#
# 执行步骤 1: 获取比特币过去一周的价格数据
# 结果: [价格数据]
#
# 执行步骤 2: 计算关键统计指标
# 结果: [统计分析]
# ...
```

### Tree of Thoughts

**思维树：探索多个推理路径**

```python
class TreeOfThoughts:
    """
    思维树：生成多个思考路径，评估并选择最佳
    """

    def __init__(self, llm, depth=3, breadth=3):
        self.llm = llm
        self.depth = depth  # 树的深度
        self.breadth = breadth  # 每层生成几个思考

    def generate_thoughts(self, state, k):
        """生成k个候选思考"""
        prompt = f"""
        当前状态：{state}

        请生成{k}个不同的下一步思考方向。

        格式：
        1. [思考1]
        2. [思考2]
        ...
        """

        response = self.llm(prompt)
        thoughts = self.parse_thoughts(response, k)
        return thoughts

    def evaluate_thought(self, thought, goal):
        """评估思考的质量"""
        prompt = f"""
        目标：{goal}
        思考：{thought}

        评估这个思考对达成目标的帮助程度（1-10分）。
        只返回数字。
        """

        score = float(self.llm(prompt).strip())
        return score

    def search(self, initial_state, goal):
        """BFS搜索最佳思考路径"""
        from collections import deque

        # 初始节点
        root = {
            "state": initial_state,
            "path": [],
            "score": 0
        }

        queue = deque([root])
        best_paths = []

        for depth in range(self.depth):
            next_queue = deque()

            while queue:
                node = queue.popleft()

                # 生成候选思考
                thoughts = self.generate_thoughts(
                    node["state"],
                    self.breadth
                )

                # 评估每个思考
                for thought in thoughts:
                    score = self.evaluate_thought(thought, goal)

                    new_node = {
                        "state": thought,
                        "path": node["path"] + [thought],
                        "score": node["score"] + score
                    }

                    next_queue.append(new_node)

            # 保留得分最高的节点
            queue = deque(sorted(
                next_queue,
                key=lambda x: x["score"],
                reverse=True
            )[:self.breadth])

            best_paths.extend(queue)

        # 返回最佳路径
        best = max(best_paths, key=lambda x: x["score"])
        return best["path"]

# 示例
tot = TreeOfThoughts(llm=gpt4, depth=3, breadth=3)

result = tot.search(
    initial_state="需要设计一个推荐系统",
    goal="设计高效、准确的推荐系统"
)

# 输出最佳思考路径
for i, thought in enumerate(result):
    print(f"步骤{i+1}: {thought}")
```

---

## 记忆管理

### 短期记忆（对话历史）

```python
class ConversationMemory:
    """对话记忆管理"""

    def __init__(self, max_tokens=4000):
        self.messages = []
        self.max_tokens = max_tokens

    def add_message(self, role, content):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })

        # 限制token数量
        self.trim_if_needed()

    def trim_if_needed(self):
        """裁剪历史以适应上下文窗口"""
        total_tokens = self.count_tokens()

        while total_tokens > self.max_tokens and len(self.messages) > 1:
            # 保留系统消息，删除最早的用户/助手消息
            for i in range(len(self.messages)):
                if self.messages[i]["role"] != "system":
                    self.messages.pop(i)
                    break

            total_tokens = self.count_tokens()

    def get_messages(self):
        return self.messages

    def summarize_old_messages(self):
        """总结旧对话"""
        if len(self.messages) > 10:
            old_messages = self.messages[:8]

            summary_prompt = f"""
            总结以下对话的关键信息：

            {old_messages}

            总结：
            """

            summary = llm(summary_prompt)

            # 替换旧消息为总结
            self.messages = [
                {"role": "system", "content": f"历史对话总结：{summary}"}
            ] + self.messages[8:]
```

### 长期记忆（向量数据库）

```python
class LongTermMemory:
    """长期记忆：基于向量数据库"""

    def __init__(self, collection_name="agent_memory"):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def store(self, content, metadata=None):
        """存储记忆"""
        from openai import OpenAI
        client = OpenAI()

        # 生成embedding
        embedding = client.embeddings.create(
            input=content,
            model="text-embedding-3-small"
        ).data[0].embedding

        # 存储
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata or {}],
            ids=[str(uuid.uuid4())]
        )

    def retrieve(self, query, k=5):
        """检索相关记忆"""
        from openai import OpenAI
        client = OpenAI()

        # 查询embedding
        query_embedding = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding

        # 检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results["documents"][0]

# 示例
memory = LongTermMemory()

# 存储知识
memory.store(
    "用户喜欢科幻小说，最喜欢的作者是阿西莫夫",
    metadata={"type": "preference", "user_id": "123"}
)

memory.store(
    "上次推荐的《基地》系列，用户很满意",
    metadata={"type": "feedback", "user_id": "123"}
)

# 检索
relevant_memories = memory.retrieve("推荐书籍")
# 返回：["用户喜欢科幻小说...", "上次推荐的《基地》..."]
```

### 工作记忆（任务状态）

```python
class WorkingMemory:
    """工作记忆：当前任务的临时状态"""

    def __init__(self):
        self.current_task = None
        self.subtasks = []
        self.completed_steps = []
        self.intermediate_results = {}

    def set_task(self, task):
        """设置当前任务"""
        self.current_task = task
        self.subtasks = []
        self.completed_steps = []
        self.intermediate_results = {}

    def add_subtask(self, subtask):
        self.subtasks.append({
            "description": subtask,
            "status": "pending",
            "result": None
        })

    def complete_subtask(self, index, result):
        """标记子任务完成"""
        if index < len(self.subtasks):
            self.subtasks[index]["status"] = "completed"
            self.subtasks[index]["result"] = result
            self.completed_steps.append(self.subtasks[index])

    def store_result(self, key, value):
        """存储中间结果"""
        self.intermediate_results[key] = value

    def get_context(self):
        """获取当前上下文"""
        return {
            "task": self.current_task,
            "completed": len([s for s in self.subtasks if s["status"] == "completed"]),
            "total": len(self.subtasks),
            "last_result": self.completed_steps[-1]["result"] if self.completed_steps else None,
            "intermediate_results": self.intermediate_results
        }
```

---

## 多Agent协作

### AutoGen框架

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# 定义多个Agent
coder = AssistantAgent(
    name="Coder",
    system_message="你是一个Python程序员，负责编写代码。",
    llm_config={"model": "gpt-4"}
)

reviewer = AssistantAgent(
    name="Reviewer",
    system_message="你是代码审查员，负责检查代码质量。",
    llm_config={"model": "gpt-4"}
)

executor = UserProxyAgent(
    name="Executor",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"}
)

# 创建群聊
groupchat = GroupChat(
    agents=[coder, reviewer, executor],
    messages=[],
    max_round=10
)

manager = GroupChatManager(groupchat=groupchat)

# 启动协作
executor.initiate_chat(
    manager,
    message="写一个函数计算斐波那契数列"
)

# 协作过程：
# Executor → Coder: "写一个函数计算斐波那契数列"
# Coder: [生成代码]
# Reviewer: [审查代码，提出改进建议]
# Coder: [修改代码]
# Executor: [执行代码，测试]
# Executor: "任务完成"
```

### 角色分工（MetaGPT）

```python
class SoftwareCompany:
    """模拟软件公司的多Agent系统"""

    def __init__(self):
        self.product_manager = ProductManager()
        self.architect = Architect()
        self.engineer = Engineer()
        self.qa = QATester()

    def develop_software(self, requirements):
        """软件开发流程"""

        # 1. 产品经理：编写PRD
        prd = self.product_manager.write_prd(requirements)
        print(f"PRD: {prd}")

        # 2. 架构师：设计系统
        design = self.architect.design_system(prd)
        print(f"设计文档: {design}")

        # 3. 工程师：实现代码
        code = self.engineer.write_code(design)
        print(f"代码: {code}")

        # 4. QA：测试
        test_report = self.qa.test(code, prd)
        print(f"测试报告: {test_report}")

        # 5. 迭代（如果测试不通过）
        while not test_report["passed"]:
            bugs = test_report["bugs"]
            code = self.engineer.fix_bugs(code, bugs)
            test_report = self.qa.test(code, prd)

        return code

class ProductManager:
    def write_prd(self, requirements):
        prompt = f"""
        作为产品经理，根据以下需求编写PRD（产品需求文档）：

        需求：{requirements}

        PRD应包括：
        1. 功能列表
        2. 用户故事
        3. 验收标准
        """
        return llm(prompt)

# 使用
company = SoftwareCompany()
result = company.develop_software("开发一个待办事项应用")
```

### Agent通信协议

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"

@dataclass
class Message:
    """Agent间的消息"""
    type: MessageType
    sender: str
    receiver: str  # "*" 表示广播
    content: Any
    metadata: dict = None

class MessageBus:
    """消息总线：Agent间通信"""

    def __init__(self):
        self.agents = {}
        self.message_queue = []

    def register_agent(self, name, agent):
        """注册Agent"""
        self.agents[name] = agent

    def send(self, message: Message):
        """发送消息"""
        if message.receiver == "*":
            # 广播
            for name, agent in self.agents.items():
                if name != message.sender:
                    agent.receive(message)
        else:
            # 点对点
            if message.receiver in self.agents:
                self.agents[message.receiver].receive(message)

    def broadcast(self, sender, content):
        """广播消息"""
        message = Message(
            type=MessageType.BROADCAST,
            sender=sender,
            receiver="*",
            content=content
        )
        self.send(message)

# Agent基类
class CommunicatingAgent:
    def __init__(self, name, message_bus):
        self.name = name
        self.message_bus = message_bus
        self.inbox = []

        # 注册到消息总线
        message_bus.register_agent(name, self)

    def send_message(self, receiver, content, msg_type=MessageType.REQUEST):
        """发送消息"""
        message = Message(
            type=msg_type,
            sender=self.name,
            receiver=receiver,
            content=content
        )
        self.message_bus.send(message)

    def receive(self, message: Message):
        """接收消息"""
        self.inbox.append(message)
        self.process_message(message)

    def process_message(self, message: Message):
        """处理消息（子类实现）"""
        pass
```

---

## 延伸阅读

**论文：**
1. ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023)
2. Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)
3. Tree of Thoughts: Deliberate Problem Solving with LLMs (Yao et al., 2023)
4. AutoGen: Enabling Next-Gen LLM Applications (Wu et al., 2023)
5. MetaGPT: Meta Programming for Multi-Agent Systems (Hong et al., 2023)

**框架：**
- LangChain Agents
- AutoGen (Microsoft)
- MetaGPT
- CrewAI
- Agency Swarm

**工具库：**
- LangChain Tools
- LlamaIndex Tools
- Transformers Agents

**下一步：**
- [多模态AI](09_Multimodal_AI.md)
- [生产部署](10_Production_Deployment.md)
