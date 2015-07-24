/* Compile with:
   clang -c -g -arch x86_64h -arch x86_64 -arch i386 fat-test.c
   libtool -static -o libfat-test.a fat-test.o

   To reduce the size of the fat .o:
   lipo -thin i386 -o fat-test.i386.o fat-test.o 
   lipo -thin x86_64 -o fat-test.x86_64.o fat-test.o 
   lipo -thin x86_64h -o fat-test.x86_64h.o fat-test.o 
   lipo -create -arch x86_64h fat-test.x86_64h.o -arch x86_64 fat-test.x86_64.o -arch i386 fat-test.i386.o -o fat-test.o -segalign i386 8 -segalign x86_64 8 -segalign x86_64h 8
 */
#ifdef __x86_64h__
int x86_64h_var;
#elif defined(__x86_64__)
int x86_64_var;
#else
int i386_var;
#endif
