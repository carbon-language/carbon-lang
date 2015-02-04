for (int c1 = 3; c1 < 2 * M; c1 += 1) {
  for (int c3 = c1 / 2 + 2; c3 <= M; c3 += 1)
    for (int c7 = c1 / 2 + 1; c7 < min(c1, c3); c7 += 1)
      S1(c7, c1 - c7, c3);
  for (int c3 = max(1, -M + c1); c3 < (c1 + 1) / 2; c3 += 1)
    S2(c1 - c3, c3);
  for (int c3 = c1 / 2 + 2; c3 <= M; c3 += 1)
    for (int c7 = c1 / 2 + 1; c7 < min(c1, c3); c7 += 1)
      S3(c7, c1 - c7, c3);
}
