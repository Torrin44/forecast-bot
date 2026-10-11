import pytest

from bot import parsing


@pytest.mark.parametrize(
    "text,expected",
    [
        ("blah\nProbability: 23%", 0.23),
        ("**Probability:** 7.5 %", 0.075),
        ("I first thought Probability: 40% but then...\nProbability: 35%", 0.35),
        ("Final answer\nprobability = 3%", 0.03),
        ("... so I land at 62%", 0.62),
    ],
)
def test_binary(text, expected):
    assert parsing.parse_binary(text) == pytest.approx(expected)


def test_binary_rejects_garbage():
    with pytest.raises(parsing.ParseError):
        parsing.parse_binary("no numbers here")


def test_percentiles_with_commas_units_and_suffix():
    text = "\n".join(
        [
            "Percentile 5: 1,200",
            "Percentile 10: $1,500",
            "Percentile 20: 1.8k",
            "Percentile 30: 2,100",
            "Percentile 40: 2,300",
            "Percentile 50: 2,500",
            "Percentile 60: 2,700",
            "Percentile 70: 2,950",
            "Percentile 80: 3,300",
            "Percentile 90: 3,900",
            "Percentile 95: 4,400",
        ]
    )
    pct = parsing.parse_percentiles(text)
    assert pct[0.2] == 1800
    assert pct[0.1] == 1500
    assert pct[0.95] == 4400


def test_percentiles_negative_and_unsorted_repair():
    text = "Percentile 5: -3.5\nPercentile 10: -2\nPercentile 20: -1\nPercentile 30: -0.5\nPercentile 40: 0.2\nPercentile 50: 0.1\nPercentile 60: 0.6\nPercentile 70: 1\nPercentile 80: 1.5\nPercentile 90: 2.4\nPercentile 95: 3"
    pct = parsing.parse_percentiles(text)
    vals = [pct[k] for k in sorted(pct)]
    assert vals == sorted(vals)
    assert pct[0.05] == -3.5


def test_percentiles_last_block_wins():
    draft = "\n".join(f"Percentile {p}: {p}" for p in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
    final = "\n".join(f"Percentile {p}: {p * 2}" for p in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
    assert parsing.parse_percentiles(draft + "\n\nRevised:\n" + final)[0.5] == 100


def test_multiple_choice_markdown_and_order():
    options = ["Joe Biden", "Donald Trump", "Other / none"]
    text = "Reasoning...\n**Joe Biden**: 10%\n- Donald Trump: 85%\nOther / none: 5%"
    dist = parsing.parse_multiple_choice(text, options)
    assert dist == pytest.approx({"Joe Biden": 0.10, "Donald Trump": 0.85, "Other / none": 0.05})


def test_multiple_choice_prefix_collision():
    options = ["0", "1", "10 or more"]
    text = "0: 50%\n1: 30%\n10 or more: 20%"
    dist = parsing.parse_multiple_choice(text, options)
    assert dist["10 or more"] == pytest.approx(0.2)
    assert dist["1"] == pytest.approx(0.3)


def test_multiple_choice_normalises():
    options = ["A", "B"]
    dist = parsing.parse_multiple_choice("A: 30%\nB: 30%", options)
    assert dist == pytest.approx({"A": 0.5, "B": 0.5})
