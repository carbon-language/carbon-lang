cert-err52-cpp
==============

The C standard library facilities setjmp() and longjmp() can be used to
simulate throwing and catching exceptions. However, these facilities bypass
automatic resource management and can result in undefined behavior, commonly
including resource leaks, and denial-of-service attacks.

This check corresponds to the CERT C++ Coding Standard rule
`ERR52-CPP. Do not use setjmp() or longjmp()
<https://www.securecoding.cert.org/confluence/pages/viewpage.action?pageId=1834>`_.
