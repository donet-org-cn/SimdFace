using System.Globalization;
using System.Text.Json;
using SimdFace;
using SimdFace.Imaging;

const string usage = """
    SimdFace 0.1.0
    simdface detect <image> [--models <directory>] [--score <0..1>] [--backend cpu|vulkan]
    simdface compare <first> <second> [--models <directory>] [--score <0..1>] [--threshold <-1..1>] [--backend cpu|vulkan]
    """;

if (args.Length == 0 || args is ["--help" or "-h"])
{
    Console.WriteLine(usage);
    return 0;
}

try
{
    bool compare = args[0] == "compare";
    int imageCount = compare ? 2 : 1;
    if ((!compare && args[0] != "detect") || args.Length <= imageCount ||
        args.Skip(1).Take(imageCount).Any(arg => arg.StartsWith("--", StringComparison.Ordinal)))
        throw new ArgumentException(usage);

    string models = "models";
    string backend = "cpu";
    float score = FaceDetector.DefaultScoreThreshold;
    float threshold = FaceRecognizer.ReferenceThreshold;
    for (int i = imageCount + 1; i < args.Length; i += 2)
    {
        if (i + 1 == args.Length)
            throw new ArgumentException($"Missing value for {args[i]}.");
        string value = args[i + 1];
        switch (args[i])
        {
            case "--models": models = value; break;
            case "--backend" when value is "cpu" or "vulkan": backend = value; break;
            case "--backend": throw new ArgumentException("--backend must be cpu or vulkan.");
            case "--score": score = ReadNumber(value, "--score", 0, 1, excludeMinimum: true); break;
            case "--threshold" when compare: threshold = ReadNumber(value, "--threshold", -1, 1); break;
            default: throw new ArgumentException($"Unknown option: {args[i]}");
        }
    }

    using var vulkan = backend == "vulkan" ? new VulkanInference() : null;
    var detector = new FaceDetector(Path.Combine(models, FaceDetector.ModelFileName), vulkan);
    var first = ImageDecoder.ReadFile(args[1]);
    var firstFaces = detector.Detect(first.Rgb, first.Width, first.Height, score);
    if (!compare)
    {
        WriteJson(new { first.Width, first.Height, Faces = firstFaces });
        return 0;
    }

    RequireOneFace(args[1], firstFaces.Count);
    var second = ImageDecoder.ReadFile(args[2]);
    var secondFaces = detector.Detect(second.Rgb, second.Width, second.Height, score);
    RequireOneFace(args[2], secondFaces.Count);

    var recognizer = new FaceRecognizer(Path.Combine(models, FaceRecognizer.ModelFileName), vulkan);
    var firstEmbedding = recognizer.Embed(first.Rgb, first.Width, first.Height, firstFaces[0].Landmarks);
    var secondEmbedding = recognizer.Embed(second.Rgb, second.Width, second.Height, secondFaces[0].Landmarks);
    float similarity = FaceRecognizer.CosineSimilarity(firstEmbedding, secondEmbedding);
    WriteJson(new { Similarity = similarity, Threshold = threshold, IsMatch = similarity >= threshold });
    return 0;
}
catch (Exception error) when (error is ArgumentException or IOException or InvalidDataException or
    UnauthorizedAccessException or NotSupportedException or InvalidOperationException)
{
    Console.Error.WriteLine(error.Message);
    return 2;
}

static float ReadNumber(string text, string option, float min, float max, bool excludeMinimum = false)
{
    if (!float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float number) ||
        !float.IsFinite(number) || number < min || number > max || (excludeMinimum && number == min))
        throw new ArgumentException($"Invalid value for {option}: {text}");
    return number;
}

static void RequireOneFace(string path, int count)
{
    if (count != 1)
        throw new InvalidDataException($"Expected exactly one face but found {count}: {path}");
}

static void WriteJson<T>(T result) => Console.WriteLine(JsonSerializer.Serialize(result, JsonSerializerOptions.Web));
