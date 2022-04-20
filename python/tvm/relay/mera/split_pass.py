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
import tvm
from tvm import relay
from tvm.relay.op.annotation import compiler_begin, compiler_end


@relay.transform.function_pass(opt_level=0)
class SplitAnnotator:
    def __init__(self, compiler):
        self.compiler = compiler
        self.visited = []
        self.merge = {}
        self.merge_level = {}
        self.previous_call = None

    def transform_function(self, func, mod, ctx):
        annotator = self

        def is_qnn_add(call):
            return (
                isinstance(call.op, tvm.relay.Function)
                and hasattr(call.op.attrs, "Composite")
                and (
                    call.op.attrs.Composite == "mera.qnn.add_relu"
                    or call.op.attrs.Composite == "mera.qnn.add"
                )
            )

        class Annotator(tvm.relay.ExprMutator):
            def visit_call(self, call):
                if call in annotator.visited:
                    return super().visit_call(call)

                if is_qnn_add(call) and call not in annotator.visited:
                    if (
                        annotator.previous_call is not None
                        and annotator.previous_call.op.name == "annotation.compiler_end"
                    ):
                        return super().visit_call(call)

                    if call in annotator.merge:
                        root_call = annotator.merge[call]
                        annotator.merge_level[root_call] += 1
                    else:
                        root_call = call
                        annotator.merge_level[root_call] = 1

                    annotator.visited.append(call)
                    new_args = []
                    for arg in call.args:
                        annotator.merge[arg] = root_call
                        new_arg = super().visit(arg)
                        new_args.append(new_arg)
                    new_call = relay.Call(call.op, new_args, call.attrs, call.type_args)

                    if annotator.merge_level[root_call] == 1:
                        annotator.merge_level[root_call] -= 1
                        ended_call = compiler_end(new_call, annotator.compiler)
                        return compiler_begin(ended_call, annotator.compiler)
                    else:
                        annotator.merge_level[root_call] -= 1
                        return new_call
                else:
                    annotator.previous_call = call
                    return super().visit_call(call)

        return Annotator().visit(func)
