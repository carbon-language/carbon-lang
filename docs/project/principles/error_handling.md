# Principles: Error handling

<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Principles

### Programming errors are not recoverable

The Carbon language and standard library will not use recoverable
error-reporting mechanisms to report programming errors, i.e. errors that are
caused by incorrect user code. Furthermore, Carbon's design will not prioritize
use cases involving recovery from programming errors.

Recovering from an error generally consists of discarding any state that might
be invalidated by the original cause of the error, and then transferring control
to a point that doesn't depend on the discarded state. For example, a function
that reads data from a file and validates a checksum might avoid modifying any
nonlocal state until validation is successful, and return early if validation
fails. This recovery strategy relies on the fact that the likely causes of the
failure are known and bounded (probably a malformed input file or an I/O error),
which allows us to put a bound on the state that might have been invalidated.

However, when a programming error is detected, the original cause is neither
known nor bounded. For example, if a function dereferences a dangling pointer,
that might mean that the author of the function forgot to check some condition
before dereferencing, or that the caller incorrectly passed a dangling pointer,
or that some other code released the memory too early, among many other
possibilities. Consequently, the only way to (mostly) reliably recover from a
programming error is to discard the entire address space and terminate the
program.

Thus, we expect that supporting recovery from programming errors would provide
little or no benefit. Furthermore, it would be harmful to several of Carbon's
primary goals:

- [Performance-critical software](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md#performance-critical-software):
  It would impose a pervasive performance overhead, because recoverable error
  handling is never free, and a programming error can occur anywhere.
- [Code that is easy to read, understand, and write](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
  Because potential programming errors are pervasive, they would have to
  propagate invisibly, which makes code harder to understand (see
  [below](#recoverable-errors-are-explicit-at-the-callsite)).
- [Software and language evolution](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md#both-software-and-language-evolution):
  It would inhibit evolution of Carbon libraries, and the Carbon language, by
  preventing them from changing how they respond to incorrect code.
- [Practical safety guarantees and testing mechanisms](https://github.com/jonmeow/carbon-lang/blob/proposal-goals/docs/project/goals.md#practical-safety-guarantees-and-testing-mechanisms):
  Similarly, it would prevent Carbon users from choosing different
  performance/safety tradeoffs for handling programming errors: if an
  out-of-bounds array access is required to throw an exception, users can't
  disable bounds checks, regardless of their risk tolerance, because code might
  rely on those exceptions being thrown.

#### Examples

If Carbon supports assertions and/or contract checking, failed assertions will
not throw exceptions, even as an optional build mode. Assertion failures will
only be presented in ways that don't alter the program state, such as logging,
terminating the program, or trapping into a debugger.

### Memory exhaustion is not recoverable

The Carbon standard library will not ordinarily use recoverable error-reporting
mechanisms to report memory exhaustion, or support user-defined code that does.

Memory exhaustion is not a programming error, and it is feasible to write code
that can successfully recover from it. However, the available evidence indicates
that very little C++ code actually does so correctly (see e.g. section 4.3 of
[this paper](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0709r4.pdf)),
which suggests that very little C++ code actually needs to do so, and we see no
reason to expect Carbon's users to differ in this respect.

Supporting recovery from memory exhaustion would impose many of the same harms
as supporting recovery from programming errors, and for the same basic reason:
memory allocation is pervasive, and so a mechanism for recovering from it would
have to be similarly pervasive. Furthermore, experience with C++ has shown that
attempting to support memory exhaustion can seriously deform the design of an
API.

#### Examples

The `pop` operation on a Carbon queue will return the value removed from the
queue. This is in contrast to C++'s `std::queue::pop()`, which does not return
the value popped from the queue, because
[that would not be exception-safe](https://isocpp.org/blog/2016/06/quick-q-why-doesnt-stdqueuepop-return-value)
due to the possibility of an out-of-memory error while copying that value.
Instead, the user must first examine the front of the queue, and then pop it as
a separate operation. Not only is this awkward for users, it means that
concurrent queues cannot match the API of non-concurrent queues (because
separate `front()` and `pop()` calls would create a race condition).

#### Caveats

Carbon will probably provide a low-level way to allocate heap memory that makes
allocation failure recoverable, because doing so appears to have few drawbacks.
However, users may need to build their own libraries on top of it, rather that
relying on the Carbon standard library, if they want to take advantage of it.
There probably will not be a way to recover from _stack_ exhaustion, because
there is no known way of doing that without major drawbacks, and users who can't
tolerate crashing due to stack overflow can normally prevent it via static
analysis.

### Recoverable errors are explicit in function declarations

Carbon functions that can emit recoverable errors will always be explicitly
marked in all function declarations.

The possibility of emitting recoverable errors is nearly as fundamental to a
function's API as its return type, and so Carbon APIs will be substantially
clearer to read, and safer to use, if we require consistent, compiler-checked
documentation of that property. Furthermore, as noted above, the mechanisms for
emitting a recoverable error always impose some performance overhead, so the
compiler must be able to distinguish the functions that need that overhead from
the ones that do not.

The default should be that functions do not emit errors, because that's the
simpler and more efficient behavior, and we also expect it to be the common
case.

#### Caveats

This principle applies only to the mechanisms that Carbon natively provides for
error recovery. We cannot guarantee that errors are explicitly marked if they
are conveyed via some other mechanism (e.g. `errno`-style thread-local storage).

### Recoverable errors are explicit at the callsite

Operations that can emit recoverable errors will always be explicitly marked at
the point of use.

If errors can propagate silently (as with exceptions in most languages), it
creates control flow paths that are not visible to the reader of the code, and
it is extremely difficult to reason about procedural code when you aren't aware
of all control flow paths. This would make Carbon code harder to understand,
maintain, and debug.

Conversely, if errors can be silently ignored (as with error return codes in
many languages), it creates a major risk of accidentally resuming normal
execution without actually recovering from the error (i.e. discarding
invalidated state). This, too, would make it extremely difficult to reason
correctly about Carbon code.

Either possibility would also allow code to evolve in unsafe ways. Changing a
function to allow it to emit errors is semantically a breaking change: client
code must now contend with a previously-impossible failure case. Requiring
errors to be marked at the callsite ensures that this breakage manifests at
build time.

#### Caveats

We cannot prevent errors from being ignored if they are conveyed via a
nonstandard mechanism (e.g. `errno`).

### Error propagation must be straightforward

Carbon will provide a means to propagate recoverable errors from any function
call to the caller of the enclosing function, with minimal textual overhead.

In our experience, it is very common for C++ code to propagate errors across
multiple layers of the call stack. C++ exceptions support this natively, and
programmers in environments without exceptions usually develop a lightweight way
to propagate errors explicitly, typically via a macro containing a conditional
`return`. In some cases they even resort to using nonstandard language
extensions in order to be able to use this operation within expressions, rather
than only at the statement level.

Given the ubiquity of this use case, Carbon must provide support for it that can
be used without altering the structure of the code, or making the non-error-case
logic less clear.

### No universal error categories

Carbon will not establish an error hierarchy or other reusable error vocabulary,
and will not prioritize use cases that involve branching based on the properties
of a propagated error.

Some languages attempt to impose a hierarchy or some other global classification
scheme for errors, in order to allow code to respond differently to different
kinds of errors, even after the errors have propagated some distance from the
function that originally raised them. However, this practice tends to be quite
brittle, because it almost inevitably requires relying on implementation
details: if a function's contract gives different meanings to different errors
it emits, it generally can't satisfy that contract by blindly propagating errors
from the functions it calls. Conversely, if it doesn't have such a contract, its
callers normally can't differentiate among the errors it emits without depending
on its implementation details.

It may make sense to distinguish certain categories of errors, if any layer of
the stack can in principle respond to those errors, and the appropriate response
requires only local knowledge. For example, any layer of the stack can respond
to an out-of-memory error by e.g. releasing any unused caches. Similarly, any
layer of the stack can respond to thread cancellation by ceasing any new
computational work and propagating the signal _even if_ it could otherwise
continue despite a failiure at that point.

However, such cases are caught between the horns of a dilemma: any error that's
universal enough to be meaningful across arbitrary levels of the call stack is
likely to be too pervasive for explicitly-marked propagation to be tolerable.
Both of the above examples have that problem; we've already ruled out
propagating out-of-memory errors because of their pervasiveness, and
cancellation is likely to pose similar challenges (although cancellation can be
ignored, which may simplify the problem somewhat).

It is certainly possible to structure a codebase so that you can reliably
propagate errors across multiple layers of the stack (so long as you control
those layers), and Carbon will support those use cases. However, it will do so
as a byproduct of general-purpose programming facilities such as pattern
matching; Carbon will not provide a separate sugar syntax for pattern-matching
error metadata, especially if that syntax can encompass multiple
potentially-failing operations. For example, if Carbon supports `try`/`catch`
statements, they will always have a single `catch` block, which will be invoked
for any error that escapes the `try` block.

## Other resources

Several other groups of language designers have arrived at similar principles.
See e.g. Swift's
[error handling rationale](https://github.com/apple/swift/blob/master/docs/ErrorHandlingRationale.rst),
[Joe Duffy's account](http://joeduffyblog.com/2016/02/07/the-error-model) of
Midori's error model, and Herb Sutter's
[pending proposal](http://wg21.link/P0709) for a new approach to exceptions in
C++.
