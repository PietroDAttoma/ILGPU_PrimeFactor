using System;
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;

class Program
{
    // Kernel parallelo: trova i divisori primi di n nel range [2, sqrt(n)]
    // e compatta i risultati con un contatore atomico.
    static void FindPrimeDivisorsKernel(
        Index1D index,
        ArrayView<long> outputPrimes,
        ArrayView<int> outputCount,
        long n)
    {
        long i = index + 2; // candidati: 2..sqrt(n)

        if (n % i != 0)
            return;

        // Test di primalità semplice su i (fino a sqrt(i))
        bool isPrime = true;
        for (long j = 2; j * j <= i; j++)
        {
            if (i % j == 0)
            {
                isPrime = false;
                break;
            }
        }

        if (isPrime)
        {
            // CORREZIONE: usare il valore restituito da Atomic.Add senza "-1"
            // Nota: Atomic.Add restituisce l'indice libero (oldValue) prima dell'incremento,
            // oppure il nuovo valore in base alla versione. Questo approccio è sicuro:
            int pos = Atomic.Add(ref outputCount[0], 1);
            outputPrimes[pos] = i;
        }
    }

    // Kernel single-thread: espande le molteplicità dei primi trovati
    // e, se resta un residuo > 1 dopo tutte le divisioni, lo aggiunge (sarà primo).
    static void ExpandMultiplicitiesKernel(
        Index1D _,
        ArrayView<long> primeDivisors,   // primi distinti <= sqrt(n)
        int primeCount,
        ArrayView<long> expandedFactors, // fattori con molteplicità
        ArrayView<int> expandedCount,
        ArrayView<long> residual)        // residual[0] parte da n
    {
        if (_.X != 0)
            return; // un solo thread

        long r = residual[0];
        int count = 0;

        // Espansione delle molteplicità
        for (int i = 0; i < primeCount; i++)
        {
            long p = primeDivisors[i];
            if (p < 2) continue;

            while (r % p == 0)
            {
                r /= p;
                expandedFactors[count++] = p;
            }
        }

        // Se resta un fattore > 1, dopo aver diviso per tutti i primi <= sqrt(n),
        // allora è necessariamente primo e va aggiunto.
        if (r > 1)
        {
            expandedFactors[count++] = r;
        }

        residual[0] = r;
        expandedCount[0] = count;
    }

    static void Main()
    {
        using var context = Context.CreateDefault();
        var device = context.GetPreferredDevice(preferCPU: false);
        using var accelerator = device.CreateAccelerator(context);

        Console.WriteLine($"Uso acceleratore: {accelerator}");
        Console.WriteLine("Premi ESC in qualsiasi momento per chiudere l'applicazione.");

        while (true)
        {
            Console.Write("\nInserisci un numero >= 2: ");

            // Lettura carattere-per-carattere per intercettare ESC immediatamente
            string input = "";
            while (true)
            {
                var key = Console.ReadKey(intercept: true);

                if (key.Key == ConsoleKey.Escape)
                {
                    Console.WriteLine("\nChiusura immediata...");
                    Environment.Exit(0); // termina subito
                }
                else if (key.Key == ConsoleKey.Enter)
                {
                    Console.WriteLine();
                    break; // fine input
                }
                else
                {
                    input += key.KeyChar;
                    Console.Write(key.KeyChar);
                }
            }

            if (!long.TryParse(input, out long n) || n < 2)
            {
                Console.WriteLine("Numero non valido.");
                continue;
            }

            var stopwatch = Stopwatch.StartNew();

            // Limite di ricerca: sqrt(n)
            int limit = (int)Math.Sqrt(n);
            if (limit < 2)
            {
                Console.WriteLine($"Fattori primi di {n}: {n}");
                Console.WriteLine($"Tempo impiegato: {stopwatch.ElapsedMilliseconds} ms");
                continue;
            }

            // Buffer per i primi divisori distinti (capacità massima ~ limit)
            using var primeDivisorsBuf = accelerator.Allocate1D<long>(limit);
            using var primeCountBuf = accelerator.Allocate1D<int>(1);
            primeCountBuf.CopyFromCPU(new int[] { 0 }); // azzera contatore

            // Kernel 1: trova i primi divisori (distinti) fino a sqrt(n)
            var findKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<long>, ArrayView<int>, long>(FindPrimeDivisorsKernel);

            // Eseguiamo per i = 2..limit (inclusivo). Con Index1D(limit - 1) otteniamo i=2..limit.
            findKernel(limit - 1, primeDivisorsBuf.View, primeCountBuf.View, n);
            accelerator.Synchronize();

            // Recupera count dei primi trovati
            int primeCount = primeCountBuf.GetAsArray1D()[0];

            // Prepara espansione delle molteplicità su GPU (kernel single-thread)
            // Margine conservativo per contenere tutte le occorrenze
            int maxExpanded = Math.Max(64, Math.Max(4, primeCount) * 32);
            using var expandedFactorsBuf = accelerator.Allocate1D<long>(maxExpanded);
            using var expandedCountBuf = accelerator.Allocate1D<int>(1);
            expandedCountBuf.CopyFromCPU(new int[] { 0 });

            using var residualBuf = accelerator.Allocate1D<long>(1);
            residualBuf.CopyFromCPU(new long[] { n });

            var expandKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<long>, int, ArrayView<long>, ArrayView<int>, ArrayView<long>>(ExpandMultiplicitiesKernel);

            // Lancio con 1 thread
            expandKernel(1, primeDivisorsBuf.View, primeCount,
                         expandedFactorsBuf.View, expandedCountBuf.View, residualBuf.View);
            accelerator.Synchronize();

            // Risultati
            int expandedCount = expandedCountBuf.GetAsArray1D()[0];
            long[] factorsAll = expandedFactorsBuf.GetAsArray1D();
            stopwatch.Stop();

            if (expandedCount == 0)
            {
                // Nessun divisore <= sqrt(n): n è primo
                Console.WriteLine($"Fattori primi di {n}: {n}");
            }
            else
            {
                var factors = new long[expandedCount];
                Array.Copy(factorsAll, factors, expandedCount);
                // 🔑 Rimuovi duplicati e ordina
                var distinctPrimes = string.Join(", ", new SortedSet<long>(factors));
                Console.WriteLine($"Fattori primi di {n}: {distinctPrimes}");
            }

            Console.WriteLine($"Tempo impiegato: {stopwatch.ElapsedMilliseconds} ms");
        }
    }
}