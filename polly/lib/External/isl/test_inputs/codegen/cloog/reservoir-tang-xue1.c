for (int c1 = 0; c1 <= 9; c1 += 2)
  for (int c3 = 0; c3 <= min(4, c1 + 3); c3 += 2)
    for (int c5 = max(1, c1); c5 <= min(c1 + 1, c1 - c3 + 4); c5 += 1)
      for (int c7 = max(1, -c1 + c3 + c5); c7 <= min(4, -c1 + c3 + c5 + 1); c7 += 1)
        S1(c1 / 2, (-c1 + c3) / 2, -c1 + c5, -c3 + c7);
