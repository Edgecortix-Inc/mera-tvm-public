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
if(USE_MERADNA_CODEGEN STREQUAL "ON")
  file(GLOB EC_RELAY_CONTRIB_SRC src/relay/backend/contrib/mera/*.cc)
  list(APPEND COMPILER_SRCS ${EC_RELAY_CONTRIB_SRC})
  file(GLOB EC_CONTRIB_SRC src/runtime/contrib/mera/*.cc)
  list(APPEND RUNTIME_SRCS ${EC_CONTRIB_SRC})
  include_directories(src/relay/backend/contrib/mera/include)
  if(MERADNA_RUNTIME_ONLY)
    set(LIBMERADNA_NAME "libmeradna-runtime.${MERADNA_HOST_ARCH}.so")
  else()
    set(LIBMERADNA_NAME "libmeradna-full.${MERADNA_HOST_ARCH}.so")
  endif()

  # Import MERA dna libraries
  file(COPY "$ENV{MERA_HOME}/lib/${LIBMERADNA_NAME}"
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FOLLOW_SYMLINK_CHAIN)
  file(GLOB MERADNA_RT_LIBS "$ENV{MERA_HOME}/lib/libdna-ip-runtime-*.so")
  file(COPY ${MERADNA_RT_LIBS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FOLLOW_SYMLINK_CHAIN)
  if(NOT MERADNA_RUNTIME_ONLY)
    file(GLOB MERADNA_TR_LIBS "$ENV{MERA_HOME}/lib/libdna-ip-tr*.so")
    file(COPY ${MERADNA_TR_LIBS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FOLLOW_SYMLINK_CHAIN)
  endif()

  set(TVM_LINKER_LIBS ${TVM_LINKER_LIBS} ${LIBMERADNA_NAME})
  set(TVM_RUNTIME_LINKER_LIBS ${TVM_RUNTIME_LINKER_LIBS} ${LIBMERADNA_NAME})

  message(STATUS "Build with Mera codegen")
endif()
