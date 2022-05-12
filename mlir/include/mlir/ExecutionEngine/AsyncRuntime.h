//===- AsyncRuntime.h - Async runtime reference implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares basic Async runtime API for supporting Async dialect
// to LLVM dialect lowering.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_ASYNCRUNTIME_H_
#define MLIR_EXECUTIONENGINE_ASYNCRUNTIME_H_

#include <stdint.h>

#ifdef mlir_async_runtime_EXPORTS
// We are building this library
#define MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS
#endif // mlir_async_runtime_EXPORTS

namespace mlir {
namespace runtime {

//===----------------------------------------------------------------------===//
// Async runtime API.
//===----------------------------------------------------------------------===//

// Runtime implementation of `async.token` data type.
using AsyncToken = struct AsyncToken;

// Runtime implementation of `async.group` data type.
using AsyncGroup = struct AsyncGroup;

// Runtime implementation of `async.value` data type.
using AsyncValue = struct AsyncValue;

// Async value payload stored in a memory owned by the async.value.
using ValueStorage = void *;

// Async runtime uses LLVM coroutines to represent asynchronous tasks. Task
// function is a coroutine handle and a resume function that continue coroutine
// execution from a suspension point.
using CoroHandle = void *;           // coroutine handle
using CoroResume = void (*)(void *); // coroutine resume function

// Async runtime uses reference counting to manage the lifetime of async values
// (values of async types like tokens, values and groups).
using RefCountedObjPtr = void *;

// Adds references to reference counted runtime object.
extern "C" void mlirAsyncRuntimeAddRef(RefCountedObjPtr, int64_t);

// Drops references from reference counted runtime object.
extern "C" void mlirAsyncRuntimeDropRef(RefCountedObjPtr, int64_t);

// Create a new `async.token` in not-ready state.
extern "C" AsyncToken *mlirAsyncRuntimeCreateToken();

// Create a new `async.value` in not-ready state. Size parameter specifies the
// number of bytes that will be allocated for the async value storage. Storage
// is owned by the `async.value` and deallocated when the async value is
// destructed (reference count drops to zero).
extern "C" AsyncValue *mlirAsyncRuntimeCreateValue(int64_t);

// Create a new `async.group` in empty state.
extern "C" AsyncGroup *mlirAsyncRuntimeCreateGroup(int64_t size);

extern "C" int64_t mlirAsyncRuntimeAddTokenToGroup(AsyncToken *, AsyncGroup *);

// Switches `async.token` to ready state and runs all awaiters.
extern "C" void mlirAsyncRuntimeEmplaceToken(AsyncToken *);

// Switches `async.value` to ready state and runs all awaiters.
extern "C" void mlirAsyncRuntimeEmplaceValue(AsyncValue *);

// Switches `async.token` to error state and runs all awaiters.
extern "C" void mlirAsyncRuntimeSetTokenError(AsyncToken *);

// Switches `async.value` to error state and runs all awaiters.
extern "C" void mlirAsyncRuntimeSetValueError(AsyncValue *);

// Returns true if token is in the error state.
extern "C" bool mlirAsyncRuntimeIsTokenError(AsyncToken *);

// Returns true if value is in the error state.
extern "C" bool mlirAsyncRuntimeIsValueError(AsyncValue *);

// Returns true if group is in the error state (any of the tokens or values
// added to the group are in the error state).
extern "C" bool mlirAsyncRuntimeIsGroupError(AsyncGroup *);

// Blocks the caller thread until the token becomes ready.
extern "C" void mlirAsyncRuntimeAwaitToken(AsyncToken *);

// Blocks the caller thread until the value becomes ready.
extern "C" void mlirAsyncRuntimeAwaitValue(AsyncValue *);

// Blocks the caller thread until the elements in the group become ready.
extern "C" void mlirAsyncRuntimeAwaitAllInGroup(AsyncGroup *);

// Returns a pointer to the storage owned by the async value.
extern "C" ValueStorage mlirAsyncRuntimeGetValueStorage(AsyncValue *);

// Executes the task (coro handle + resume function) in one of the threads
// managed by the runtime.
extern "C" void mlirAsyncRuntimeExecute(CoroHandle, CoroResume);

// Executes the task (coro handle + resume function) in one of the threads
// managed by the runtime after the token becomes ready.
extern "C" void mlirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *, CoroHandle,
                                                     CoroResume);

// Executes the task (coro handle + resume function) in one of the threads
// managed by the runtime after the value becomes ready.
extern "C" void mlirAsyncRuntimeAwaitValueAndExecute(AsyncValue *, CoroHandle,
                                                     CoroResume);

// Executes the task (coro handle + resume function) in one of the threads
// managed by the runtime after the all members of the group become ready.
extern "C" void
mlirAsyncRuntimeAwaitAllInGroupAndExecute(AsyncGroup *, CoroHandle, CoroResume);

// Returns the current number of available worker threads in the threadpool.
extern "C" int64_t mlirAsyncRuntimGetNumWorkerThreads();

//===----------------------------------------------------------------------===//
// Small async runtime support library for testing.
//===----------------------------------------------------------------------===//

extern "C" void mlirAsyncRuntimePrintCurrentThreadId();

} // namespace runtime
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_ASYNCRUNTIME_H_
