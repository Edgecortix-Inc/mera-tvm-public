/*
 * Copyright 2022 EdgeCortix Inc.
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

def clean(build_dir) {
    sh "rm -rf ${build_dir}"
}

def build(build_dir, jobs, libmeradna_name) {
    sh "mkdir -p ${build_dir}"
    sh """
        cd ${build_dir} && cmake -DUSE_OPENMP=gnu -DUSE_LLVM=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
        make -j ${jobs}
    """
}

def buildRuntime(build_dir, jobs, libmeradna_name) {
    sh "mkdir -p ${build_dir}"
    sh """
        cd ${build_dir} && cmake -DUSE_OPENMP=gnu -DUSE_LLVM=OFF -DUSE_LIBBRACKTRACE=OFF -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DMERADNA_RUNTIME_ONLY=ON ..
        make runtime -j ${jobs}
    """
}

return this
