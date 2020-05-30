<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Carbon generics: motivating use cases

## **Intro**

This reflects the use cases we set out to address with Carbon's generics and template systems. These are "things we'd like to represent" and "properties of the result we'd like to achieve" and not about mechanisms or programming models.


## **Motivating use cases**

The majority of motivating use cases are those underpinning the existing C++ template system and the generics systems used in languages like Rust and Swift. We don't enumerate all of these here in detail, but a core set of use cases for Carbon are of course the existing C++ code idioms in modern and clean C++ codebases, including:



*   Containers (arrays, maps, lists)
*   Algorithms (sort, search)
*   Wrappers (similar to optional, variant)
*   [Policy-based design](https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design#Policy-based_design) (need to be judicious, these are often misused)
*   Configurable / parametric APIs such as the storage customized `std::chrono` APIs
*   Compile-time polymorphism
*   Parameterized numeric types (`std::complex&lt;T>`)
*   MAYBE: Something like expression templates for compile-time code inlining / generation
*   TODO: please add more that I'm forgetting

However, there are some use cases that are very specific to Carbon that we want to highlight, as they help motivate specific aspects of the rest of the design.

**A: Incrementally restricting template code**

The entire idea of statically typed languages is that coding against specific types and interfaces is a better model and experience. Unfortunately, templates don't provide many of those benefits to programmers until it's too late (users are consuming the API) and with high overhead (template error messages). Generally, code should move towards more rigorously type checked constructs. However, existing C++ code is full of unrestricted usage of duck-typed templates. They are incredibly convenient to write and so likely will continue to exist for a long time.

As a consequence, one of the motivating use cases for Carbon's design is to support template-style code, but give an easy and very incremental path for increasing strictness and adding type checking to the code. To this end, we want to support the following incremental evolution sequence for code:



1. Initially written using raw duck-typed templates.
2. Extract a single type parameter into a local generic "interface" description which uses templates to provide a default implementation. The original template is then adjusted to code against this interface rather than the raw duck-typed interface. Other type parameters would remain templated.
3. Update users to explicitly implement the generic interface rather than relying on the default.
4. Remove the default implementation, requiring users to supply implementations.
5. Repeat steps #2 - #4 for each type, interface, or parameter until fully based on specific generic interfaces.

**B: Incrementally erasing or monomorphizing generic interfaces**

Templates (and equally generics) often cause significant code size scaling issues. A classic technique to address this is to use type erasure and perform runtime dynamic dispatch instead of compile-time monomorphization. Moving between generics and these erased constructs should be both a trivial change and an incremental change.



*   Each generic interface should be erasable independent of others.
*   When an interface supports it, erasure shouldn't change code for users of the interface.
*   Erasure should use compiler-optimization friendly techniques to minimize its overhead and make it more available even under performance constraints.
    *   Example: arrange the representation to make any function pointers easily analyzable across call edges. 
*   Migration ease should be bi-directional to allow incrementally expanding the generic interface.

**C: Non-inheritance support for basic abstract interface & dependency injection**

When migrating C++ code to Carbon, programmers should be able to reliably convert C++ abstract base classes where there is a single level of hierarchy (every derived class is final) to a type erased interface that doesn't involve inheritance. The resulting code should not be any more complex, and should have clean paths to more powerful systems such as non-erased generics.

Similarly, types which only support subclassing for test stubs and mocks (basically merging the normal implementation into the interface, as used in ["Dependency injection"](https://en.wikipedia.org/wiki/Dependency_injection)) should be able to easily migrate to a more principled form of type erasure and dependency injection.

Last but not least, we should support automatic construction of loosely coupled interfaces as found in Go without reaching for custom infrastructure.

**D: Runtime Polymorphism (or Value Semantics and Concept-based Polymorphism)**

See:



*   [https://sean-parent.stlab.cc/papers-and-presentations/#better-code-runtime-polymorphism](https://sean-parent.stlab.cc/papers-and-presentations/#better-code-runtime-polymorphism)
*   [https://youtu.be/QGcVXgEVMJg](https://youtu.be/QGcVXgEVMJg)

These are specific C++ idioms advocated by Sean Parent and others and widely deployed. They require significant boilerplate and complexity in C++. Carbon should provide natural language facilities that directly support these patterns. Ideally, C++ code written carefully within these idiomatic lines can be directly migrated to the Carbon constructs, and the Carbon constructs exposed using the same API surface as the C++ idioms.
