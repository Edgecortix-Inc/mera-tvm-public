# Copyright 2022 EdgeCortix Inc.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from typing import List
from ..expr import Call
from ..function import Function
from ..expr_functor import ExprVisitor
from ..transform import transform
from ...ir.op import Op

def get_consumers(src : Call, full_call : Call) -> List[Call]:
    """Get a list of all calls that are consumers of call 'src' in the 'full_call' code.
    :param src: Call we want to extract the consumers
    :return: A list containing all the calls that have 'src' as one of their arguments in 'full_call'.
    """
    class FindConsumers(ExprVisitor):
        def __init__(self, call, base_call):
            ExprVisitor.__init__(self)
            self.call = call
            self.base_call = base_call
            self.consumers = []

        def __is_op(self, call, op):
            return isinstance(call, Call) and isinstance(call.op, Op) and call.op.name == op

        def visit_call(self, call):
            if not self.__is_op(call, 'layout_transform'):
                for arg in call.args:
                    if self.__is_op(arg, 'layout_transform'):
                        arg = arg.args[0]
                    if arg == self.call:
                        self.consumers.append(call)
            return super().visit_call(call)

        def visit_tuple(self, tup):
            for arg in tup.fields:
                if self.__is_op(arg, 'layout_transform'):
                    arg = arg.args[0]
                if arg == self.call:
                    self.consumers = FindConsumers(tup, self.base_call).get_consumers()
            return super().visit_tuple(tup)

        def get_consumers(self):
            self.visit(self.base_call)
            return self.consumers
    return FindConsumers(src, full_call).get_consumers()


def get_consumers_mc(src : Call) -> List[Call]:
    """Get a list of all calls that are consumers of call 'src'. This function is meant to be called
    during MergeComposite phase, as it will fetch the current MergeComposite code as the full code list.

    :param src: Call we want to extract the consumers
    :return: A list containing all the calls that have 'src' as one of their arguments during MergeComposite.
    """
    curr_mc = transform.GetCurrentMergeComposite()
    # Check if the 'src' call is output of the graph
    if curr_mc.body == src:
        return [curr_mc.body]
    return get_consumers(src, curr_mc)

def find_call_of_type(fcn : Call, op_name : str) -> List[Call]:
    """Searches through a function call for calls whose operators are 'op_name' and returns a list with all of them.
    This is meant to be used to traverse through Functions flattening them and fetching the relevant calls during merge
    composite checking.

    :param fcn: Call that will serve as the source of the search tree. If it's not a function just checks if the current call
    is of Op 'op_name'. If it's a function, recursively traverses through it collecting the inner calls that match.
    :param op_name: Name of the operator to look for
    :return: A list of all the collected calls, empty list if none have matched.
    """
    class NodeFinder(ExprVisitor):
        def __init__(self, op_name):
            ExprVisitor.__init__(self)
            self.op_name = op_name
            self.collected = []

        def visit_call(self, call):
            if call.op.name == self.op_name:
                self.collected.append(call)
            return super().visit_call(call)

        def find(self, fcn):
            self.visit(fcn)
            return self.collected
    assert isinstance(fcn, Call)
    if not isinstance(fcn.op, Function):
        return [fcn] if fcn.op.name == op_name else []
    return NodeFinder(op_name).find(fcn.op)
