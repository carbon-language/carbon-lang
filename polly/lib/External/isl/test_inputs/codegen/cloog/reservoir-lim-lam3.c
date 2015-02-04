for (int c1 = 5; c1 <= 5 * M; c1 += 1) {
  for (int c3 = max(2, floord(-M + c1, 4)); c3 < min(M, (c1 + 1) / 3 - 2); c3 += 1)
    for (int c5 = max(1, -M - c3 + (M + c1) / 2 - 2); c5 < min(c3, -2 * c3 + (c1 + c3) / 2 - 2); c5 += 1)
      S1(c1 - 2 * c3 - 2 * c5 - 5, c3, c5);
  for (int c3 = max(1, floord(-M + c1, 4)); c3 < (c1 + 1) / 5; c3 += 1)
    S2(c1 - 4 * c3 - 3, c3);
  if (c1 % 5 == 0)
    S4(c1 / 5);
  for (int c3 = max(-3 * M - c1 + 3 * ((M + c1) / 2) + 1, -((c1 - 1) % 3) + 3); c3 < (c1 + 1) / 5; c3 += 3)
    S3((c1 - 2 * c3 - 1) / 3, c3);
}
