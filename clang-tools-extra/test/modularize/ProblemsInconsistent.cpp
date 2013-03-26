# RUN: modularize %s -x c++ 2>&1 | FileCheck %s

InputProblemsInconsistent/Header1.h
InputProblemsInconsistent/Header2.h

# CHECK: error: 'SYMBOL' defined at both {{.*}}{{[/\\]}}InputProblemsInconsistent{{[/\\]}}SubHeader.h:3:9 and {{.*}}{{[/\\]}}InputProblemsInconsistent/SubHeader.h:6:9
# CHECK-NEXT: error: header '{{.*}}{{[/\\]}}InputProblemsInconsistent{{[/\\]}}SubHeader.h' has different contents dependening on how it was included
# CHECK-NEXT: note: 'SYMBOL' in {{.*}}{{[/\\]}}InputProblemsInconsistent{{[/\\]}}SubHeader.h at 3:9 not always provided
# CHECK-NEXT: note: 'SYMBOL' in {{.*}}{{[/\\]}}InputProblemsInconsistent{{[/\\]}}SubHeader.h at 6:9 not always provided
# CHECK-NEXT: note: 'TypeInt' in {{.*}}{{[/\\]}}InputProblemsInconsistent{{[/\\]}}SubHeader.h at 10:13 not always provided
