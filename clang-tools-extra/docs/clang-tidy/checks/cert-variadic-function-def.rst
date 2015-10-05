cert-dcl50-cpp
========================

A variadic function using an ellipsis has no mechanisms to check the type safety
of arguments being passed to the function or to check that the number of
arguments being passed matches the semantics of the function definition.
Consequently, a runtime call to a C-style variadic function that passes
inappropriate arguments yields undefined behavior. Such undefined behavior could
be exploited to run arbitrary code.

This check corresponds to the CERT C++ Coding Standard rule
`DCL50-CPP. Do not define a C-style variadic function
<https://www.securecoding.cert.org/confluence/display/cplusplus/DCL50-CPP.+Do+not+define+a+C-style+variadic+function>`_.
