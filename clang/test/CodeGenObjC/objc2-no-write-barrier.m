// RUN: clang-cc -triple x86_64-apple-darwin9 -fobjc-gc -emit-llvm -o %t %s &&
// RUN: grep 'objc_assign' %t | count 0

typedef struct {
    int ival;
    id submenu;
} XCBinderContextMenuItem;

id actionMenuForDataNode(void) {
    XCBinderContextMenuItem menusToCreate[]  = {
        {1, 0}
    };
    return 0;
}

XCBinderContextMenuItem GmenusToCreate[]  = {
        {1, 0}
};
