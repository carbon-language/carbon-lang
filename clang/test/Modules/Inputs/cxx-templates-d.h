@import cxx_templates_common;

inline int InstantiateWithAnonymousDeclsD(WithAnonymousDecls<char> x) { return (x.k ? x.a : x.b) + (x.k ? x.s.c : x.s.d) + x.e; }
