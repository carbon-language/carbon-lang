/******************************************************************************\
  Extended version of omp_set_unset_lock.c for testing hinted locks.
  Check to make sure OpenMP locks guarantee mutual 
  exclusion for multiple threads.
\******************************************************************************/
	
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void cscall(int id, int n[1000], int *passed, omp_lock_t *lock) {
    int i;

    omp_set_lock( lock );
    for (i = 0; i < 1000; i++) {
        n[i] = id;
    }
    for (i = 0; i < 1000; i++) {
        if ( n[i] != id ) {
            *passed = 0;
        }
    }
    omp_unset_lock( lock );
}

int hinted_lock(kmp_lock_hint_t hint) {
    int passed, n[1000], j, id;
    omp_lock_t lock;
    
    passed = 1;

    kmp_init_lock_hinted(&lock, hint);

    #pragma omp parallel shared(n, passed, lock) private(id, j)	
    {
        id = omp_get_thread_num();
        for (j = 1; j <= 10000; j++) {
            cscall( id, n, &passed, &lock );
        }
    }

    omp_destroy_lock(&lock);

    if (passed) {
        return 0;
    } else {
        return 1;
    }
}

int main() {
    int ret = 0;
    ret += hinted_lock(kmp_lock_hint_none);
    ret += hinted_lock(kmp_lock_hint_contended);
    ret += hinted_lock(kmp_lock_hint_uncontended);
    ret += hinted_lock(kmp_lock_hint_nonspeculative);
    ret += hinted_lock(kmp_lock_hint_speculative);
    // This one will emit Warning on machines with no TSX. 
    ret += hinted_lock(kmp_lock_hint_adaptive);
    if (ret) {
        printf(" Test %s failed\n", __FILE__);
        return 1;
    } else {
        printf(" Test %s passed\n", __FILE__);
        return 0;
    }
}
