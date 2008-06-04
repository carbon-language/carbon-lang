// RUN: clang -emit-llvm < %s

// PR2414
typedef void filter_func_t();
filter_func_t mono_filter;

void addfilter2(filter_func_t *func){}

void setup_filters()
{
        addfilter2( mono_filter);
}

