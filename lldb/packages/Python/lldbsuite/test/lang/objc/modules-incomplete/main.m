@import Foundation;
@import myModule;

int main()
{
    @autoreleasepool
    {
        MyClass *myObject = [MyClass alloc];
        [myObject publicMethod]; // Set breakpoint 0 here.
    }
}
