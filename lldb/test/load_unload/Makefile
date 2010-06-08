all: a.out liba.dylib libb.dylib libc.dylib

CFLAGS ?=-arch x86_64 -gdwarf-2 -O0

a.out: main.o
	gcc $(CFLAGS) -o a.out main.o

main.o: main.c
	gcc $(CFLAGS) -c main.c

liba.dylib: a.o libb.dylib
	gcc $(CFLAGS) -dynamiclib -install_name "@executable_path/liba.dylib" -o liba.dylib a.o -L. -lb
	dsymutil liba.dylib

a.o: a.c
	gcc $(CFLAGS) -c a.c

libb.dylib: b.o
	gcc $(CFLAGS) -dynamiclib -install_name "@executable_path/libb.dylib" -o libb.dylib b.o
	dsymutil libb.dylib

b.o: b.c
	gcc $(CFLAGS) -c b.c

libc.dylib: c.o
	gcc $(CFLAGS) -dynamiclib -install_name "@executable_path/libc.dylib" -o libc.dylib c.o
	dsymutil libc.dylib

c.o: c.c
	gcc $(CFLAGS) -c c.c

clean:
	rm -rf *.o *~ *.dylib a.out *.dSYM
