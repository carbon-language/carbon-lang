; RUN: opt -disable-output -passes=no-op-module,no-op-module %s
; RUN: opt -disable-output -passes='module(no-op-module,no-op-module)' %s
