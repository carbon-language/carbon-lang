# RUN: modularize %s -x c++ 2>&1 | FileCheck %s

InputProblemsDuplicate/Header1.h
InputProblemsDuplicate/Header2.h

# CHECK: error: 'TypeInt' defined at both {{.*}}{{[/\\]}}InputProblemsDuplicate{{[/\\]}}Header1.h:2:13 and {{.*}}{{[/\\]}}InputProblemsDuplicate{{[/\\]}}Header2.h:2:13
