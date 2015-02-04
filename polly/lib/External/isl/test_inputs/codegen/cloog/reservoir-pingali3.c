{
  for (int c1 = 1; c1 <= M; c1 += 1)
    for (int c3 = 1; c3 <= M; c3 += 1)
      S1(c1, c3);
  for (int c1 = 1; c1 <= M; c1 += 1)
    for (int c3 = 1; c3 <= M; c3 += 1)
      for (int c5 = 1; c5 <= M; c5 += 1)
        S2(c1, c3, c5);
}
