// Test is line- and column-sensitive. See run lines below.

template<typename T, T Value, template<typename U, U ValU> class X>
void f(X<T, Value> x);

// RUN: c-index-test -test-load-source all %s | FileCheck -check-prefix=CHECK-LOAD %s
// CHECK-LOAD: index-templates.cpp:4:6: FunctionTemplate=f:4:6 Extent=[3:1 - 4:22]
// CHECK-LOAD: index-templates.cpp:3:19: TemplateTypeParameter=T:3:19 (Definition) Extent=[3:19 - 3:20]
// CHECK-LOAD: index-templates.cpp:3:24: NonTypeTemplateParameter=Value:3:24 (Definition) Extent=[3:22 - 3:29]
// FIXME: Need the template type parameter here
// CHECK-LOAD: index-templates.cpp:3:66: TemplateTemplateParameter=X:3:66 (Definition) Extent=[3:31 - 3:67]
// CHECK-LOAD: index-templates.cpp:4:20: ParmDecl=x:4:20 (Definition) Extent=[4:8 - 4:21]
// FIXME: Need the template declaration here.
// FIXME: Need the template type parameter here
// CHECK-LOAD: index-templates.cpp:4:13: DeclRefExpr=Value:3:24 Extent=[4:13 - 4:18]

// RUN: c-index-test -test-load-source-usrs all %s | FileCheck -check-prefix=CHECK-USRS %s
// CHECK-USRS: index-templates.cpp c:@FT@>3#T#Nt0.0#t>2#T#Nt1.0f#>t0.22t0.0# Extent=[3:1 - 4:22]
// CHECK-USRS: index-templates.cpp c:index-templates.cpp@79 Extent=[3:19 - 3:20]
// CHECK-USRS: index-templates.cpp c:index-templates.cpp@82 Extent=[3:22 - 3:29]
// CHECK-USRS: index-templates.cpp c:index-templates.cpp@91 Extent=[3:31 - 3:67]
// CHECK-USRS: index-templates.cpp c:index-templates.cpp@136@FT@>3#T#Nt0.0#t>2#T#Nt1.0f#>t0.22t0.0#@x Extent=[4:8 - 4:21]
