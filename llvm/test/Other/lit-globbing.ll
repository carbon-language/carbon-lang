RUN: echo XXA > %T/XXA.txt
RUN: echo XXB > %T/XXB.txt
RUN: echo XXAB > %T/XXAB.txt

RUN: echo %T/XXA* | FileCheck -check-prefix=STAR %s
RUN: echo %T/'XXA'* | FileCheck -check-prefix=STAR %s
RUN: echo %T/XX'A'* | FileCheck -check-prefix=STAR %s

RUN: echo %T/XX?.txt | FileCheck -check-prefix=QUESTION %s
RUN: echo %T/'XX'?.txt | FileCheck -check-prefix=QUESTION %s

RUN: echo %T/XX??.txt | FileCheck -check-prefix=QUESTION2 %s
RUN: echo %T/'XX'??.txt | FileCheck -check-prefix=QUESTION2 %s

RUN: echo 'XX*' 'XX?.txt' 'XX??.txt' | FileCheck -check-prefix=QUOTEDARGS %s

STAR-NOT: XXB.txt
STAR: {{(XXA.txt.*XXAB.txt|XXAB.txt.*XXA.txt)}}

QUESTION-NOT: XXAB.txt
QUESTION: {{(XXA.txt.*XXB.txt|XXB.txt.*XXA.txt)}}

QUESTION2-NOT: XXA.txt
QUESTION2-NOT: XXB.txt
QUESTION2: XXAB.txt

QUOTEDARGS-NOT: .txt
QUOTEDARGS: XX* XX?.txt XX??.txt
