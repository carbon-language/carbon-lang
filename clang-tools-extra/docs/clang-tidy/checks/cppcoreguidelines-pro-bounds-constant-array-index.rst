cppcoreguidelines-pro-bounds-constant-array-index
=================================================

This check flags all array subscriptions on static arrays and std::arrays that either have a non-compile-time constant index or are out of bounds (for std::array).
For out-of-bounds checking of static arrays, see the clang-diagnostic-array-bounds check.

Dynamic accesses into arrays are difficult for both tools and humans to validate as safe. gsl::span is a bounds-checked, safe type for accessing arrays of data. gsl::at() is another alternative that ensures single accesses are bounds-checked. If iterators are needed to access an array, use the iterators from an gsl::span constructed over the array.

The check can generated fixes after the option cppcoreguidelines-pro-bounds-constant-array-index.GslHeader has been set to the name of the
include file that contains gsl::at(), e.g. "gsl/gsl.h".

This rule is part of the "Bounds safety" profile of the C++ Core Guidelines, see
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#-bounds2-only-index-into-arrays-using-constant-expressions.
