// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

typedef struct { } rwlock_t;
struct fs_struct { rwlock_t lock; int umask; };
void __copy_fs_struct(struct fs_struct *fs) { fs->lock = (rwlock_t) { }; }

