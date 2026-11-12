using System.Diagnostics;
using System.Security.Cryptography;

namespace SimdFace;

/// <summary>记录被测人的自然表情基线，并用它判定随机选中的眼睛或嘴部动作。</summary>
/// <param name="blink">为 <see langword="true"/> 时检查闭眼和睁眼，为 <see langword="false"/> 时检查张嘴和合嘴。</param>
internal sealed class FacialAction(bool blink)
{
    private const float MinOpenEye = 0.10f;
    private const float MaxClosedMouth = 0.15f;
    private static readonly TimeSpan HoldTime = TimeSpan.FromMilliseconds(120);
    private static readonly TimeSpan EyeClosureHold = TimeSpan.FromMilliseconds(65);

    private enum Phase { Neutral, Action, Return }
    private Phase phase;
    private long? heldSince;
    private float baselineA;
    private float baselineB;

    /// <summary>是否已采集到睁眼基线，进入闭眼或重新睁眼阶段。</summary>
    public bool IsBlinkInProgress => blink && phase != Phase.Neutral;
    /// <summary>是否正在等待双眼闭合的样本。</summary>
    public bool WaitingForEyeClosure => blink && phase == Phase.Action;

    /// <summary>以相同概率选择眼睛动作或嘴部动作。</summary>
    public static FacialAction CreateRandom() => new(RandomNumberGenerator.GetInt32(2) == 0);

    /// <summary>当前阶段要展示的动作指令。</summary>
    public FaceExpressionPrompt Prompt => (blink, phase) switch
    {
        (_, Phase.Neutral) => FaceExpressionPrompt.Neutral,
        (true, Phase.Action) => FaceExpressionPrompt.CloseEyes,
        (true, Phase.Return) => FaceExpressionPrompt.OpenEyes,
        (false, Phase.Action) => FaceExpressionPrompt.OpenMouth,
        _ => FaceExpressionPrompt.CloseMouth
    };

    /// <summary>根据一帧人脸网格，采集自然表情基线或推进动作阶段。</summary>
    /// <param name="points">由 <see cref="FaceMesh.Estimate"/> 返回的 468 个交错排列的 XYZ 坐标。</param>
    /// <param name="now">处理本帧时由 <see cref="Stopwatch.GetTimestamp"/> 取得的时间戳。</param>
    /// <returns>完成指定动作并稳定恢复自然表情后返回 <see langword="true"/>。</returns>
    public bool Observe(ReadOnlySpan<float> points, long now)
    {
        // 分别用眼宽和嘴宽归一化上下眼睑与内唇之间的距离。
        float a, b;
        if (blink)
        {
            a = (Distance(points, 160, 144) + Distance(points, 158, 153)) / (2 * Distance(points, 33, 133));
            b = (Distance(points, 385, 380) + Distance(points, 387, 373)) / (2 * Distance(points, 362, 263));
        }
        else
        {
            a = b = Distance(points, 13, 14) / Distance(points, 78, 308);
        }

        bool qualifies = (blink, phase) switch
        {
            (true, Phase.Neutral) => a >= MinOpenEye && b >= MinOpenEye,
            (true, Phase.Action) => EyeClosed(a, baselineA) && EyeClosed(b, baselineB),
            (true, Phase.Return) => a >= baselineA * 0.82f && b >= baselineB * 0.82f,
            (false, Phase.Neutral) => a <= MaxClosedMouth,
            (false, Phase.Action) => a >= MathF.Max(0.24f, baselineA + 0.13f),
            _ => a <= MathF.Min(MaxClosedMouth, baselineA + 0.05f)
        };
        if (!qualifies || !float.IsFinite(a) || !float.IsFinite(b))
        {
            heldSince = null;
            return false;
        }
        // 快速眨眼时可能只采到一帧明显的双眼闭合。
        // 闭眼前和重新睁眼后都需要稳定的睁眼画面。
        bool clearlyClosed = WaitingForEyeClosure && a <= baselineA * 0.58f && b <= baselineB * 0.58f;
        if (!clearlyClosed)
        {
            if (heldSince is not long start)
            {
                heldSince = now;
                if (blink && phase == Phase.Neutral)
                {
                    baselineA = a;
                    baselineB = b;
                }
                return false;
            }
            if (Stopwatch.GetElapsedTime(start, now) < (WaitingForEyeClosure ? EyeClosureHold : HoldTime))
                return false;
        }

        heldSince = null;
        if (phase == Phase.Return)
            return true;
        if (phase == Phase.Neutral)
        {
            baselineA = blink ? (baselineA + a) / 2 : a;
            baselineB = blink ? (baselineB + b) / 2 : b;
        }
        phase++;
        return false;
    }

    /// <summary>重置当前动作的进度，等待重新采集自然表情基线。</summary>
    public void Reset()
    {
        phase = Phase.Neutral;
        heldSince = null;
    }

    // 对每只眼分别检查开合比相对基线的下降比例和下降幅度。
    private static bool EyeClosed(float current, float baseline) =>
        current <= MathF.Min(baseline * 0.76f, baseline - 0.025f);

    private static float Distance(ReadOnlySpan<float> points, int a, int b)
    {
        float x = points[a * 3] - points[b * 3];
        float y = points[a * 3 + 1] - points[b * 3 + 1];
        return MathF.Sqrt(x * x + y * y);
    }
}
