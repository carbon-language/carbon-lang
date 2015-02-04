{
  S1();
  S2();
  for (int c1 = 0; c1 < N; c1 += 1)
    for (int c3 = 0; c3 < N; c3 += 1) {
      S4(c1, c3);
      S5(c1, c3);
    }
  for (int c1 = 0; c1 < N; c1 += 1)
    for (int c3 = 0; c3 < N; c3 += 1)
      for (int c5 = 0; c5 <= (N - 1) / 32; c5 += 1) {
        S7(c1, c3, c5, 32 * c5);
        for (int c7 = 32 * c5 + 1; c7 <= min(N - 1, 32 * c5 + 31); c7 += 1) {
          S6(c1, c3, c5, c7 - 1);
          S7(c1, c3, c5, c7);
        }
        if (32 * c5 + 31 >= N) {
          S6(c1, c3, c5, N - 1);
        } else
          S6(c1, c3, c5, 32 * c5 + 31);
      }
  S8();
}
