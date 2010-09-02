// RUN: %llvmgcc %s -S -m32 -o - | FileCheck %s
// Formerly this generated code that did a load past the end of the structure.
// That was fixed by 46726, but that patch had bad side effects and was
// reverted.  This has been fixed another way in the meantime.
extern void abort();
extern void exit(int);
struct T
{
unsigned i:8;
unsigned c:24;
};
f(struct T t)
{
struct T s[1];
s[0]=t;
return(char)s->c;
}
main()
{
// CHECK:  getelementptr inbounds %struct.T* %t, i32 0, i32 0 
// CHECK:  getelementptr inbounds %struct.T* %t, i32 0, i32 0
struct T t;
t.i=0xff;
t.c=0xffff11;
if(f(t)!=0x11)abort();
exit(0);
}
