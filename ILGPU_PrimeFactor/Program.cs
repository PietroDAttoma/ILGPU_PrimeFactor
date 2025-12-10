using System;
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;

class Program
{
    // Kernel GPU: trova divisori primi di n
    static void FullGpuFactorKernel(Index1D index, ArrayView<long> results, long n)
    {
        long i = index + 2; // partiamo da 2
        if (n % i == 0)
        {
            // Test di primalità in GPU
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
                results[index] = i;
            else
                results[index] = 0;
        }
        else
        {
            results[index] = 0;
        }
    }

    static void Main()
    {
        Console.Write("Inserisci un numero intero >= 2: ");
        if (!long.TryParse(Console.ReadLine(), out long n) || n < 2)
        {
            Console.WriteLine("Numero non valido.");
            return;
        }

        var stopwatch = Stopwatch.StartNew();

        using var context = Context.CreateDefault();
        var device = context.GetPreferredDevice(preferCPU: false);
        using var accelerator = device.CreateAccelerator(context);

        Console.WriteLine($"Uso acceleratore: {accelerator}");

        // Buffer per i risultati
        var buffer = accelerator.Allocate1D<long>((int)(n - 1));

        // Carico il kernel
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, long>(FullGpuFactorKernel);

        // Lancio il kernel
        kernel((int)(n - 1), buffer.View, n);
        accelerator.Synchronize();

        // Recupero i risultati
        long[] divisors = buffer.GetAsArray1D();

        stopwatch.Stop();

        // Stampa solo i valori > 0
        Console.WriteLine($"Fattori primi di {n}: {string.Join(", ", Array.FindAll(divisors, d => d > 0))}");
        Console.WriteLine($"Tempo impiegato: {stopwatch.ElapsedMilliseconds} ms");
    }
}