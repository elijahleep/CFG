import ast
import networkx as nx
import inspect
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from z3 import *
import sympy
import operator

class SimpleCFGBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_node = 0

    def add_node(self, label):
        self.graph.add_node(self.current_node, label=label)
        node_id = self.current_node
        self.current_node += 1
        return node_id

    def add_edge(self, from_node, to_node, label=None):
        self.graph.add_edge(from_node, to_node, label=label)

    def build_from_source(self, source_code):
        tree = ast.parse(source_code)
        self.current_node = 0
        self._visit(tree)
        return self.graph

    def build_from_file(self, filename):
        with open(filename, "r") as source:
            source_code = source.read()
        return self.build_from_source(source_code)

    def _visit(self, node, parent=None):
        if isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            func_node = self.add_node(f"Function: {node.name}({', '.join(args)})")
            if parent is not None:
                self.add_edge(parent, func_node)
            current_parent = func_node
        elif isinstance(node, ast.If):
            condition = self._format_condition(node.test)
            if_node = self.add_node(f"If {condition}")
            self.add_edge(parent, if_node)

            body_parent = if_node
            last_body_node = None
            for child in node.body:
                body_parent = self._visit(child, body_parent)
                last_body_node = body_parent

            else_parent = if_node
            last_else_node = if_node
            for child in node.orelse:
                else_parent = self._visit(child, else_parent)
                last_else_node = else_parent

            if last_body_node is not None:
                next_node = self.current_node
                self.add_edge(last_body_node, next_node)
            if not node.orelse:
                self.add_edge(if_node, self.current_node, label="false")
            else:
                self.add_edge(last_else_node, self.current_node)

            self.add_edge(if_node, if_node + 1, label="true")
            if node.orelse:
                self.add_edge(if_node, else_parent, label="false")

            return self.current_node - 1
        elif isinstance(node, ast.Return):
            return_node = self.add_node(f"Return {self._format_value(node.value)}")
            self.add_edge(parent, return_node)
            return return_node
        elif isinstance(node, ast.For):
            target = self._format_value(node.target)
            iter_label = self._format_value(node.iter)
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                for_node = self.add_node(f"For {target} in range({', '.join(self._format_value(arg) for arg in node.iter.args)})")
            else:
                for_node = self.add_node(f"For {target} in {iter_label}")
            self.add_edge(parent, for_node)
            current_parent = for_node

            last_body_node = None
            body_nodes = []
            for child in node.body:
                body_node = self._visit(child, current_parent)
                body_nodes.append(body_node)
                current_parent = body_node

            for i in range(len(body_nodes) - 1):
                self.add_edge(body_nodes[i], body_nodes[i + 1])

            if body_nodes:
                self.add_edge(body_nodes[-1], for_node)

            return for_node
        elif isinstance(node, ast.Assign):
            targets = ", ".join(self._format_value(t) for t in node.targets)
            value = self._format_value(node.value)
            assign_node = self.add_node(f"Assign {targets} = {value}")
            self.add_edge(parent, assign_node)
            return assign_node
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call_node = self.add_node(f"Call {self._format_value(node.value)}")
            self.add_edge(parent, call_node)
            return call_node
        else:
            current_parent = parent

        last_visited_node = current_parent
        for child in ast.iter_child_nodes(node):
            last_visited_node = self._visit(child, last_visited_node)

        return last_visited_node

    def _format_condition(self, node):
        if isinstance(node, ast.Compare):
            left = self._format_value(node.left)
            comparators = [self._format_value(c) for c in node.comparators]
            ops = [self._format_op(op) for op in node.ops]
            return f"{left} {' '.join(f'{op} {comp}' for op, comp in zip(ops, comparators))}"
        return self._format_value(node)

    def _format_value(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.BinOp):
            left = self._format_value(node.left)
            right = self._format_value(node.right)
            op = self._format_op(node.op)
            return f"({left} {op} {right})"
        elif isinstance(node, ast.Call):
            func = self._format_value(node.func)
            args = ', '.join(self._format_value(arg) for arg in node.args)
            return f"{func}({args})"
        return ast.dump(node)

    def _format_op(self, node):
        return {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
        }.get(type(node), type(node).__name__)

    def visualize(self):
        pos = graphviz_layout(self.graph, prog="dot")
        labels = nx.get_node_attributes(self.graph, 'label')
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        nx.draw(self.graph, pos, with_labels=True, labels=labels, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')
        plt.show()

class SymbolicExecution:
    def __init__(self, cfg_builder):
        self.cfg_builder = cfg_builder
        self.solver = Solver()

    def _sympy_to_z3(self, expr):
        if isinstance(expr, sympy.core.symbol.Symbol):
            return Int(str(expr))
        elif isinstance(expr, sympy.core.relational.Relational):
            left = self._sympy_to_z3(expr.lhs)
            right = self._sympy_to_z3(expr.rhs)
            if isinstance(expr, sympy.core.relational.Equality):
                return left == right
            elif isinstance(expr, sympy.core.relational.StrictGreaterThan):
                return left > right
            elif isinstance(expr, sympy.core.relational.StrictLessThan):
                return left < right
            elif isinstance(expr, sympy.core.relational.GreaterThan):
                return left >= right
            elif isinstance(expr, sympy.core.relational.LessThan):
                return left <= right
            elif isinstance(expr, sympy.core.relational.Unequality):
                return left != right
        elif isinstance(expr, sympy.core.numbers.Integer):
            return IntVal(expr)
        elif isinstance(expr, sympy.logic.boolalg.And):
            return And(*[self._sympy_to_z3(arg) for arg in expr.args])
        elif isinstance(expr, sympy.logic.boolalg.Or):
            return Or(*[self._sympy_to_z3(arg) for arg in expr.args])
        elif isinstance(expr, sympy.logic.boolalg.Not):
            return Not(self._sympy_to_z3(expr.args[0]))
        elif isinstance(expr, sympy.core.function.Mod):
            left = self._sympy_to_z3(expr.args[0])
            right = self._sympy_to_z3(expr.args[1])
            return left % right
        return expr

    def _symbolic_value(self, node):
        if isinstance(node, ast.Name):
            return sympy.Symbol(node.id)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._symbolic_value(node.left)
            right = self._symbolic_value(node.right)
            op = self._binop_to_func(node.op)
            return op(left, right)
        elif isinstance(node, ast.Call):
            func = self._symbolic_value(node.func)
            args = [self._symbolic_value(arg) for arg in node.args]
            return func(*args)
        return None

    def _binop_to_func(self, op):
        return {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
        }.get(type(op), None)

    def symbolic_execution(self, source_code):
        cfg = self.cfg_builder.build_from_source(source_code)
        solutions = []
        for node in nx.topological_sort(cfg):
            if "label" in cfg.nodes[node]:
                label = cfg.nodes[node]["label"]
                if label.startswith("If "):
                    condition = label[3:]
                    condition_expr = sympy.sympify(condition)
                    z3_condition = self._sympy_to_z3(condition_expr)

                    # Check the condition (if branch)
                    self.solver.push()
                    self.solver.add(z3_condition)
                    if self.solver.check() == sat:
                        solutions.append((label, self.solver.model()))
                    self.solver.pop()

                    # Check the negation of the condition (else branch)
                    self.solver.push()
                    self.solver.add(Not(z3_condition))
                    if self.solver.check() == sat:
                        solutions.append((f"Else: not({condition})", self.solver.model()))
                    self.solver.pop()
        return solutions

# Пример кода для символического выполнения
source_code = """
def example(x, y):
    if x > y:
        return x
    else:
        return y
"""

cfg_builder = SimpleCFGBuilder()
symbolic_executor = SymbolicExecution(cfg_builder)

# Выполнение символического выполнения
solutions = symbolic_executor.symbolic_execution(source_code)

# Вывод решений
for condition, model in solutions:
    print(f"Condition: {condition}")
    print("Model:")
    print(model)

# Визуализация CFG
cfg_builder.visualize()
