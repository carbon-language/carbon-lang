// RUN: %clang_cc1 %s
void functionPromotion(void f(char *const []));
void arrayPromotion(char * const argv[]);

int whatever(int argc, char *argv[])
{
        arrayPromotion(argv);
        functionPromotion(arrayPromotion);
}
