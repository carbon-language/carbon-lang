struct inner
{
    int var_d;
};

struct my_type
{
    int var_a;
    char var_b;
    struct inner inner_;
};

int local_struct_test(void)
{
    struct my_type var_c;
    var_c.var_a = 10;
    var_c.var_b = 'a';
    var_c.inner_.var_d = 30;
    return 0; // BP_LOCAL_STRUCT
}

int local_array_test(void)
{
    int array[3];
    array[0] = 100;
    array[1] = 200;
    array[2] = 300;
    return 0; // BP_LOCAL_ARRAY
}

int local_pointer_test(void)
{
    const char* test_str = "Rakaposhi";
    int var_e = 24;
    int *ptr = &var_e;
    return 0; // BP_LOCAL_PTR
}

int local_test()
{
    local_struct_test();
    local_array_test();
    local_pointer_test();
    return 0;
}