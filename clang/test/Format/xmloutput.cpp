// RUN: clang-format -output-replacements-xml -sort-includes %s > %t.xml
// RUN: FileCheck -strict-whitespace -input-file=%t.xml %s

// CHECK: <?xml
// CHECK-NEXT: {{<replacements.*incomplete_format='false'}}
// CHECK-NEXT: {{<replacement.*#include &lt;a>&#10;#include &lt;b><}}
// CHECK-NEXT: {{<replacement.*>&#10;<}}
// CHECK-NEXT: {{<replacement.*> <}}
#include <b>
#include <a>

int a;int*b;
