for (int c0 = 1; c0 <= N; c0 += 1) {
  S1(c0);
  for (int c1 = 1; c1 <= M; c1 += 1)
    S2(c0, c1);
}
for (int c0 = N + 1; 1; c0 += 1)
  S1(c0);
