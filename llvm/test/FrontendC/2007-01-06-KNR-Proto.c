// RUN: %llvmgcc -S -o - %s
// PR1083

int svc_register (void (*dispatch) (int));

int svc_register (dispatch)
     void (*dispatch) ();
{
}

