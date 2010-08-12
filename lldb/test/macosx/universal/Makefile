testit: testit.i386 testit.x86_64
	lipo -create -o testit testit.i386 testit.x86_64

testit.i386: testit.i386.o
	gcc -arch i386 -o testit.i386 testit.i386.o

testit.x86_64: testit.x86_64.o
	gcc -arch x86_64 -o testit.x86_64 testit.x86_64.o

testit.i386.o: main.c
	gcc -g -O0 -arch i386 -c -o testit.i386.o main.c

testit.x86_64.o: main.c
	gcc -g -O0 -arch x86_64 -c -o testit.x86_64.o main.c

clean:
	rm -rf testit* *~
