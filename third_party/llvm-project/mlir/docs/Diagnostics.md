# Diagnostic Infrastructure

[TOC]

This document presents an introduction to using and interfacing with MLIR's
diagnostics infrastructure.

See [MLIR specification](LangRef.md) for more information about MLIR, the
structure of the IR, operations, etc.

## Source Locations

Source location information is extremely important for any compiler, because it
provides a baseline for debuggability and error-reporting. The
[builtin dialect](Dialects/Builtin.md) provides several different location
attributes types depending on the situational need.

## Diagnostic Engine

The `DiagnosticEngine` acts as the main interface for diagnostics in MLIR. It
manages the registration of diagnostic handlers, as well as the core API for
diagnostic emission. Handlers generally take the form of
`LogicalResult(Diagnostic &)`. If the result is `success`, it signals that the
diagnostic has been fully processed and consumed. If `failure`, it signals that
the diagnostic should be propagated to any previously registered handlers. It
can be interfaced with via an `MLIRContext` instance.

```c++
DiagnosticEngine& engine = ctx->getDiagEngine();

/// Handle the reported diagnostic.
// Return success to signal that the diagnostic has either been fully processed,
// or failure if the diagnostic should be propagated to the previous handlers.
DiagnosticEngine::HandlerID id = engine.registerHandler(
    [](Diagnostic &diag) -> LogicalResult {
  bool should_propagate_diagnostic = ...;
  return failure(should_propagate_diagnostic);
});


// We can also elide the return value completely, in which the engine assumes
// that all diagnostics are consumed(i.e. a success() result).
DiagnosticEngine::HandlerID id = engine.registerHandler([](Diagnostic &diag) {
  return;
});

// Unregister this handler when we are done.
engine.eraseHandler(id);
```

### Constructing a Diagnostic

As stated above, the `DiagnosticEngine` holds the core API for diagnostic
emission. A new diagnostic can be emitted with the engine via `emit`. This
method returns an [InFlightDiagnostic](#inflight-diagnostic) that can be
modified further.

```c++
InFlightDiagnostic emit(Location loc, DiagnosticSeverity severity);
```

Using the `DiagnosticEngine`, though, is generally not the preferred way to emit
diagnostics in MLIR. [`operation`](LangRef.md/#operations) provides utility
methods for emitting diagnostics:

```c++
// `emit` methods available in the mlir namespace.
InFlightDiagnostic emitError/Remark/Warning(Location);

// These methods use the location attached to the operation.
InFlightDiagnostic Operation::emitError/Remark/Warning();

// This method creates a diagnostic prefixed with "'op-name' op ".
InFlightDiagnostic Operation::emitOpError();
```

## Diagnostic

A `Diagnostic` in MLIR contains all of the necessary information for reporting a
message to the user. A `Diagnostic` essentially boils down to three main
components:

*   [Source Location](#source-locations)
*   Severity Level
    -   Error, Note, Remark, Warning
*   Diagnostic Arguments
    -   The diagnostic arguments are used when constructing the output message.

### Appending arguments

One a diagnostic has been constructed, the user can start composing it. The
output message of a diagnostic is composed of a set of diagnostic arguments that
have been attached to it. New arguments can be attached to a diagnostic in a few
different ways:

```c++
// A few interesting things to use when composing a diagnostic.
Attribute fooAttr;
Type fooType;
SmallVector<int> fooInts;

// Diagnostics can be composed via the streaming operators.
op->emitError() << "Compose an interesting error: " << fooAttr << ", " << fooType
                << ", (" << fooInts << ')';

// This could generate something like (FuncAttr:@foo, IntegerType:i32, {0,1,2}):
"Compose an interesting error: @foo, i32, (0, 1, 2)"
```

### Attaching notes

Unlike many other compiler frameworks, notes in MLIR cannot be emitted directly.
They must be explicitly attached to another diagnostic non-note diagnostic. When
emitting a diagnostic, notes can be directly attached via `attachNote`. When
attaching a note, if the user does not provide an explicit source location the
note will inherit the location of the parent diagnostic.

```c++
// Emit a note with an explicit source location.
op->emitError("...").attachNote(noteLoc) << "...";

// Emit a note that inherits the parent location.
op->emitError("...").attachNote() << "...";
```

## InFlight Diagnostic

Now that [Diagnostics](#diagnostic) have been explained, we introduce the
`InFlightDiagnostic`, an RAII wrapper around a diagnostic that is set to be
reported. This allows for modifying a diagnostic while it is still in flight. If
it is not reported directly by the user it will automatically report when
destroyed.

```c++
{
  InFlightDiagnostic diag = op->emitError() << "...";
}  // The diagnostic is automatically reported here.
```

## Diagnostic Configuration Options

Several options are provided to help control and enhance the behavior of
diagnostics. These options can be configured via the MLIRContext, and registered
to the command line with the `registerMLIRContextCLOptions` method. These
options are listed below:

### Print Operation On Diagnostic

Command Line Flag: `-mlir-print-op-on-diagnostic`

When a diagnostic is emitted on an operation, via `Operation::emitError/...`,
the textual form of that operation is printed and attached as a note to the
diagnostic. This option is useful for understanding the current form of an
operation that may be invalid, especially when debugging verifier failures. An
example output is shown below:

```shell
test.mlir:3:3: error: 'module_terminator' op expects parent op 'builtin.module'
  "module_terminator"() : () -> ()
  ^
test.mlir:3:3: note: see current operation: "module_terminator"() : () -> ()
  "module_terminator"() : () -> ()
  ^
```

### Print StackTrace On Diagnostic

Command Line Flag: `-mlir-print-stacktrace-on-diagnostic`

When a diagnostic is emitted, attach the current stack trace as a note to the
diagnostic. This option is useful for understanding which part of the compiler
generated certain diagnostics. An example output is shown below:

```shell
test.mlir:3:3: error: 'module_terminator' op expects parent op 'builtin.module'
  "module_terminator"() : () -> ()
  ^
test.mlir:3:3: note: diagnostic emitted with trace:
 #0 0x000055dd40543805 llvm::sys::PrintStackTrace(llvm::raw_ostream&) llvm/lib/Support/Unix/Signals.inc:553:11
 #1 0x000055dd3f8ac162 emitDiag(mlir::Location, mlir::DiagnosticSeverity, llvm::Twine const&) /lib/IR/Diagnostics.cpp:292:7
 #2 0x000055dd3f8abe8e mlir::emitError(mlir::Location, llvm::Twine const&) /lib/IR/Diagnostics.cpp:304:10
 #3 0x000055dd3f998e87 mlir::Operation::emitError(llvm::Twine const&) /lib/IR/Operation.cpp:324:29
 #4 0x000055dd3f99d21c mlir::Operation::emitOpError(llvm::Twine const&) /lib/IR/Operation.cpp:652:10
 #5 0x000055dd3f96b01c mlir::OpTrait::HasParent<mlir::ModuleOp>::Impl<mlir::ModuleTerminatorOp>::verifyTrait(mlir::Operation*) /mlir/IR/OpDefinition.h:897:18
 #6 0x000055dd3f96ab38 mlir::Op<mlir::ModuleTerminatorOp, mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::HasParent<mlir::ModuleOp>::Impl, mlir::OpTrait::IsTerminator>::BaseVerifier<mlir::OpTrait::HasParent<mlir::ModuleOp>::Impl<mlir::ModuleTerminatorOp>, mlir::OpTrait::IsTerminator<mlir::ModuleTerminatorOp> >::verifyTrait(mlir::Operation*) /mlir/IR/OpDefinition.h:1052:29
 #  ...
  "module_terminator"() : () -> ()
  ^
```

## Common Diagnostic Handlers

To interface with the diagnostics infrastructure, users will need to register a
diagnostic handler with the [`DiagnosticEngine`](#diagnostic-engine).
Recognizing the many users will want the same handler functionality, MLIR
provides several common diagnostic handlers for immediate use.

### Scoped Diagnostic Handler

This diagnostic handler is a simple RAII class that registers and unregisters a
given diagnostic handler. This class can be either be used directly, or in
conjunction with a derived diagnostic handler.

```c++
// Construct the handler directly.
MLIRContext context;
ScopedDiagnosticHandler scopedHandler(&context, [](Diagnostic &diag) {
  ...
});

// Use this handler in conjunction with another.
class MyDerivedHandler : public ScopedDiagnosticHandler {
  MyDerivedHandler(MLIRContext *ctx) : ScopedDiagnosticHandler(ctx) {
    // Set the handler that should be RAII managed.
    setHandler([&](Diagnostic diag) {
      ...
    });
  }
};
```

### SourceMgr Diagnostic Handler

This diagnostic handler is a wrapper around an llvm::SourceMgr instance. It
provides support for displaying diagnostic messages inline with a line of a
respective source file. This handler will also automatically load newly seen
source files into the SourceMgr when attempting to display the source line of a
diagnostic. Example usage of this handler can be seen in the `mlir-opt` tool.

```shell
$ mlir-opt foo.mlir

/tmp/test.mlir:6:24: error: expected non-function type
func @foo() -> (index, ind) {
                       ^
```

To use this handler in your tool, add the following:

```c++
SourceMgr sourceMgr;
MLIRContext context;
SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
```

#### Filtering Locations

In some situations, a diagnostic may be emitted with a callsite location in a
very deep call stack in which many frames are unrelated to the user source code.
These situations often arise when the user source code is intertwined with that
of a large framework or library. The context of the diagnostic in these cases is
often obfuscated by the unrelated framework source locations. To help alleviate
this obfuscation, the `SourceMgrDiagnosticHandler` provides support for
filtering which locations are shown to the user. To enable filtering, a user
must simply provide a filter function to the `SourceMgrDiagnosticHandler` on
construction that indicates which locations should be shown. A quick example is
shown below:

```c++
// Here we define the functor that controls which locations are shown to the
// user. This functor should return true when a location should be shown, and
// false otherwise. When filtering a container location, such as a NameLoc, this
// function should not recurse into the child location. Recursion into nested
// location is performed as necessary by the caller.
auto shouldShowFn = [](Location loc) -> bool {
  FileLineColLoc fileLoc = loc.dyn_cast<FileLineColLoc>();

  // We don't perform any filtering on non-file locations.
  // Reminder: The caller will recurse into any necessary child locations.
  if (!fileLoc)
    return true;

  // Don't show file locations that contain our framework code.
  return !fileLoc.getFilename().strref().contains("my/framework/source/");
};

SourceMgr sourceMgr;
MLIRContext context;
SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context, shouldShowFn);
```

Note: In the case where all locations are filtered out, the first location in
the stack will still be shown.

### SourceMgr Diagnostic Verifier Handler

This handler is a wrapper around a llvm::SourceMgr that is used to verify that
certain diagnostics have been emitted to the context. To use this handler,
annotate your source file with expected diagnostics in the form of:

*   `expected-(error|note|remark|warning) {{ message }}`

A few examples are shown below:

```mlir
// Expect an error on the same line.
func @bad_branch() {
  cf.br ^missing  // expected-error {{reference to an undefined block}}
}

// Expect an error on an adjacent line.
func @foo(%a : f32) {
  // expected-error@+1 {{unknown comparison predicate "foo"}}
  %result = arith.cmpf "foo", %a, %a : f32
  return
}

// Expect an error on the next line that does not contain a designator.
// expected-remark@below {{remark on function below}}
// expected-remark@below {{another remark on function below}}
func @bar(%a : f32)

// Expect an error on the previous line that does not contain a designator.
func @baz(%a : f32)
// expected-remark@above {{remark on function above}}
// expected-remark@above {{another remark on function above}}

```

The handler will report an error if any unexpected diagnostics were seen, or if
any expected diagnostics weren't.

```shell
$ mlir-opt foo.mlir

/tmp/test.mlir:6:24: error: unexpected error: expected non-function type
func @foo() -> (index, ind) {
                       ^

/tmp/test.mlir:15:4: error: expected remark "expected some remark" was not produced
// expected-remark {{expected some remark}}
   ^~~~~~~~~~~~~~~~~~~~~~~~~~
```

Similarly to the [SourceMgr Diagnostic Handler](#sourcemgr-diagnostic-handler),
this handler can be added to any tool via the following:

```c++
SourceMgr sourceMgr;
MLIRContext context;
SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
```

### Parallel Diagnostic Handler

MLIR is designed from the ground up to be multi-threaded. One important to thing
to keep in mind when multi-threading is determinism. This means that the
behavior seen when operating on multiple threads is the same as when operating
on a single thread. For diagnostics, this means that the ordering of the
diagnostics is the same regardless of the amount of threads being operated on.
The ParallelDiagnosticHandler is introduced to solve this problem.

After creating a handler of this type, the only remaining step is to ensure that
each thread that will be emitting diagnostics to the handler sets a respective
'orderID'. The orderID corresponds to the order in which diagnostics would be
emitted when executing synchronously. For example, if we were processing a list
of operations [a, b, c] on a single-thread. Diagnostics emitted while processing
operation 'a' would be emitted before those for 'b' or 'c'. This corresponds 1-1
with the 'orderID'. The thread that is processing 'a' should set the orderID to
'0'; the thread processing 'b' should set it to '1'; and so on and so forth.
This provides a way for the handler to deterministically order the diagnostics
that it receives given the thread that it is receiving on.

A simple example is shown below:

```c++
MLIRContext *context = ...;
ParallelDiagnosticHandler handler(context);

// Process a list of operations in parallel.
std::vector<Operation *> opsToProcess = ...;
llvm::parallelForEachN(0, opsToProcess.size(), [&](size_t i) {
  // Notify the handler that we are processing the i'th operation.
  handler.setOrderIDForThread(i);
  auto *op = opsToProcess[i];
  ...

  // Notify the handler that we are finished processing diagnostics on this
  // thread.
  handler.eraseOrderIDForThread();
});
```
