int
sum (int a, int b)
{
    int result = a + b; // Set a breakpoint here
    asm("nop");
    return result;
}

int
main(int argc, char const *argv[])
{

    int array[3];

    array[0] = sum (1238, 78392);
    array[1] = sum (379265, 23674);
    array[2] = sum (872934, 234);

    return 0;
}
