
typedef struct {
  int op;
} event_t;

event_t test(int X) {
  event_t foo, bar;
  return X ? foo : bar;
}
