import os
import time
import torch
from transformers import AutoTokenizer, pipeline

# AUTO-DETECTION OF AVAILABLE RESOURCES
IS_GPU = torch.cuda.is_available()
CPU_CORES = os.cpu_count()

print(f"[INFO] GPU Available: {IS_GPU}")
print(f"[INFO] CPU Cores: {CPU_CORES}")


# ML SUMMARIZATION MODELS

# CPU model (fastest)
CPU_MODEL = "sshleifer/distilbart-xsum-12-6"

# GPU models
GPU_MODEL_FAST = "sshleifer/distilbart-cnn-12-6"    # default (fast)
GPU_MODEL_HIGH_QUALITY = "facebook/bart-large-cnn"  # optional (slower)

# CPU SUMMARIZER
def load_cpu_summarizer():
    optimal_threads = max(2, CPU_CORES // 2)
    os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
    os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
    print(f"[CPU] Using {optimal_threads} threads")

    tokenizer = AutoTokenizer.from_pretrained(CPU_MODEL, use_fast=True)

    summarizer = pipeline(
        "summarization",
        model=CPU_MODEL,
        tokenizer=tokenizer,
        device=-1,
        batch_size=6,
    )

    return summarizer, tokenizer


# GPU SUMMARIZER (TWO MODELS)
def load_gpu_summarizer(model_name="fast"):
    if model_name == "fast":
        model = GPU_MODEL_FAST
        print("[GPU] Using FAST model:", model)
    else:
        model = GPU_MODEL_HIGH_QUALITY
        print("[GPU] Using HIGH-QUALITY model:", model)

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=0,
        batch_size=10,
        torch_dtype=torch.float16,
    )

    return summarizer, tokenizer


# SELECT DEVICE + MODEL

# Change this to "fast" or "quality" when GPU is available
GPU_MODEL_SELECTION = "fast"  # <<< YOU CAN SWITCH THIS ANYTIME

if IS_GPU:
    summarizer, tokenizer = load_gpu_summarizer(GPU_MODEL_SELECTION)
    print("[MODE] Running on GPU")
else:
    summarizer, tokenizer = load_cpu_summarizer()
    print("[MODE] Running on CPU")



# TRUNCATION(the max tokens the model can handle is 1024)
def truncate_articles(texts, max_tokens=1000):
    encoded = tokenizer(
        texts,
        max_length=max_tokens,
        truncation=True,
        add_special_tokens=False,
    )
    return tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True)


# MAIN SUMMARIZATION FUNCTION
def summarize_articles(articles):
    print(f"[TASK] Summarizing {len(articles)} article(s)...")
    start = time.time()

    truncated = truncate_articles(articles)

    summaries = summarizer(
        truncated,
        do_sample=False,
        num_beams=3,        # fast + good quality
        length_penalty=1.0,
        min_length=40,
        max_length=140,
        clean_up_tokenization_spaces=True
    )

    total_time = time.time() - start
    print(f"[Speed] {len(articles)} articles in {total_time:.2f} sec")

    # return summaries
    return [summary["summary_text"] for summary in summaries]

if __name__ == "__main__":
    scraped_articles = [
    "Stock market today: Dow, S&P 500, Nasdaq slide out gains as Nvidia, tech stocks lead sharp reversal lower US stocks were hammered on Thursday as an abrupt midday reversal in markets followed a blockbuster earnings report from Nvidia (NVDA) and the release of the long-delayed September jobs report, which showed hiring grew more than expected in the final month of the third quarter. When the closing bell rang on Thursday, the tech-heavy Nasdaq Composite (^IXIC) had taken the sharpest losses, falling more than 2.1% from Wednesday's close, and marking a more than 3.5% intraday reversal. The S&P 500 (^GSPC) fell 1.5%, and the Dow Jones Industrial Average (^DJI) lost 380 points, or 0.8%. Stocks opened the session higher after Nvidia earnings, the September jobs report, and Walmart's strong third quarter lifted investor spirits. The abrupt turn lower in trading on Thursday came alongside a sharp drop in the crypto markets, with bitcoin and ether both falling sharply to add to recent losses. Bitcoin was trading below $87,000 at the close of stock trading on Thursday, near its lowest levels since April. Nvidia stock finished the day down 3.1% after surging as much as 5% in early trading after the chipmaker delivered an earnings beat and issued a stronger-than-expected revenue outlook for the fourth quarter. CEO Jensen Huang said demand for the company’s Blackwell processors is ",
    "Stocks fell on Thursday, as a market-wide rally sparked by blockbuster Nvidia results and guidance gave up steam and as investors lost hope that the Federal Reserve would cut rates again in December. The Dow Jones Industrial Average fell 386.51 points, or 0.84%, to settle at 45,752.26, after rallying more than 700 points at session highs. The S&P 500 shed 1.56% to end the day at 6,538.76, despite rising as much as 1.9% earlier in the day. The Nasdaq Composite fell 2.16% to finish at 22,078.05, down from a 2.6% advance at one point in the session. Nvidia's reversal dragged the broader market down. Shares had gained as much as 5% after the chipmaker released better-than-expected quarterly results and an upbeat fourth-quarter sales forecast. However, the stock ultimately closed down 3%, despite CEO Jensen Huang's reassurances that demand for its current-generation Blackwell chips are off the charts. He also rejected the idea of an AI bubble. But concerns about AI stock valuations returned, with investors also pondering the impact on the sector if the Fed doesn't lower interest rates further. Oracle and AMD were among the first AI plays to fall into the red on the session, followed by Nvidia. Adding to downward pressure was a shutdown-delayed September jobs report that showed the economy added 119,000 jobs, more than economists' expected. Following the data, the latest fed funds futures trading showed less than 40% odds that the central bank would cut for a third time this year",
    "LIVE Dow Rises Nearly 500 Points The S&P 500 and the Nasdaq Composite also rose solidly. Last Updated: Nov. 21, 2025 at 4:50 PM ET Key Events 8 hours ago Home Builders Jump on Fed Rate Cut Expectations 8 hours ago The Nasdaq Rallies Nearly 2%. Wall Street Is Finally Buying the Dip. 10 hours ago The Stock Market Rally Is Back On 11 hours ago The Nasdaq Is Rising Again. Will It Last? 12 hours ago Nasdaq Gives Back Early Gains for Second Straight Day 13 hours ago Consumer Sentiment Improved From Preliminary Reading, but Americans Are Still Gloomy 13 hours ago International Stocks Expected to Top Charts in 2026, BofA Survey Shows 13 hours ago Stocks Are Gaining as December Rate-Cut Odds Jump 13 hours ago Oil Down Amid Geopolitical Developments, U.S. Rate-Cut Uncertainty 14 hours ago Treasury Yields Fall Again Amid AI Worries, Fed Uncertainty Pinned Post Updated 7 hours ago The Dow Gains Nearly 500 Points, Wraps Choppy Week With ‘Everything Rally’ By Connor Smith Pretty much everything rallied on Friday as Wall Street shook off yesterday’s technology stock selloff. The Dow Jones Industrial Average rose 493 points, or 1.1%. The S&P 500 was up 1%. The Nasdaq Composite was up 0.9%. The yield on the 2-year Treasury note was down to 3.51%. The 10-year yield was down to 4.06%. Investors piled into stocks and bonds after New York Federal Reserve President John Williams reignited bets on a December rate cut when he made the case for another cut in the near term. He argued monetary policy ",
    "Equity performance in 2025 has been shaped by multiple crosscurrents. High levels of geopolitical uncertainty and shifting U.S. trade policy have triggered bouts of high equity volatility, while supportive fiscal agendas, easing monetary policy, strong corporate earnings and AI-related optimism have supported risk sentiment. Overall, developed market equities have posted strong double-digit gains year-to-date, sustained by a resilient macro backdrop. Watch our latest video to discover more. Developed market equities advanced as investors focused on structural drivers rather than short-term disruptions. Performance has varied by region and across quarters. Q1 saw rising investor fears over the outlook for U.S. equities, with U.S., policy both increasing trade tensions and causing debt and budget management concerns. European equities significantly outperformed on prospects of increased cross-Europe defense spending, fiscal expansion in Germany, and hopes for a Russia–Ukraine de-escalation. Q2 unfolded in two phases: an early sell-off on intensified U.S. trade rhetoric and new country tariffs announced on 2 April, followed by a sharp rebound after a 90-day pause in enacting the announced tariffs. This lifted the markets which had lagged in the previous quarter, such as the U.S. and Japan, and sectors including Technology and Communication Services. Through Q3 and into October, risk sentiment improved further with the announcement of framework trade agreements betwee",
    "5Share Save Play(12min) Comments(10) Summary The S&P 500 (SPX) is poised for an end-of-year rally, supported by renewed federal spending and the conclusion of the government shutdown. Fiscal flows surged in October, creating a $226 billion private sector surplus, historically a positive signal for risk asset prices like SPX. The Federal Reserve's shift from quantitative tightening to easing, alongside falling rates, will inject liquidity and boost both equities and real estate. Seasonal trends, fiscal acceleration, and a looser Fed stance all align for strong risk asset performance into year-end, with further upside possible from tariff relief. J Studios/DigitalVision via Getty Images J Studios/DigitalVision via Getty Images Introduction This report applies the sectoral balance framework to assess how fiscal and monetary dynamics are shaping risk asset performance. With inflation cooling, unemployment rising, and the Federal Reserve signaling a shift in policy, the interplay between fiscal This article was written by Analyst’s Disclosure:I/we have no stock, option or similar derivative position in any of the companies mentioned, and no plans to initiate any such positions within the next 72 hours. I wrote this article myself, and it expresses my own opinions. I am not receiving compensation for it. I have no business relationship with any company whose stock is mentioned in this article.Selling real estate, long gold, cash and select AI stocks Seeking Alpha's Disclosure: Pas",
    "As November 2025 unfolds, global financial markets are caught in a relentless tempest of volatility, characterized by rapid and often dramatic price swings. Investor sentiment has grown increasingly cautious, with a palpable weakening of risk appetite across asset classes. This turbulent environment is forcing a critical reassessment of investment strategies and highlighting the imperative for robust risk management in an era defined by uncertainty and swift market shifts. The immediate implication for investors is a challenging landscape where traditional certainties are being questioned. Major indices are experiencing significant daily fluctuations, signaling a broader market unease. This heightened unpredictability underscores the need for a nuanced understanding of the underlying forces at play and a disciplined approach to navigating the ups and downs of today's trading environment. Unpacking the Market's Turbulent Core: A Deep Dive into Current Dynamics The current market turmoil is far from a fleeting phenomenon; it represents a complex interplay of economic, technological, and geopolitical factors that have coalesced to create a landscape of elevated uncertainty. Specific instances underscore the severity of the present conditions: the Dow Jones Industrial Average has, on at least one occasion, shed 800 points in a single day, marking one of its most substantial drops in over a month. Similarly, both the S&P 500 and the Nasdaq have recently dipped below their critical",
    "CONCLUDED Stock Market News From Nov. 20, 2025: Nasdaq Sinks 2.2% as Nvidia Bump Evaporates The Dow, S&P 500, and Nasdaq marked their biggest blown gains since April. Last Updated: Nov. 20, 2025 at 7:17 PM ET Key Events 3 hours ago Oil Gives Up Gains on U.S. Peace Efforts for Russia-Ukraine 3 hours ago Dollar Bounces Back from Payrolls-Triggered Decline 4 hours ago The Stock Market’s Fear Gauge Is Starting to Flash 5 hours ago The Stock Market Is on Track for Its Biggest Blown Gain Since April 6 hours ago Bonds Get Tailwind From Late Jobs Report 7 hours ago The Stock Market Gives Back Early Gains as Nvidia Turns Lower 8 hours ago What Tech Selloff? S&P 500, Nasdaq Turn Positive for the Week 9 hours ago Stocks Are Surging. Nvidia Gives the AI Rally New Life. 9 hours ago Dollar Turns Lower After U.S. Jobs Data 10 hours ago December Rate-Cut Odds Get a Slight Lift Pinned Post 3 hours ago Nasdaq Falls 2.2%, Gives Up Early 2.6% Gain, as Nvidia Rally Fizzles By Connor Smith An early stock market rally fueled by Nvidia's earnings turned out to be a mirage on Thursday. The Nasdaq Composite sank 2.2%. The S&P 500 fell 1.6%. The Dow Jones Industrial Average dropped 386 points, or 0.8%. At one point, the Nasdaq was up 2.6%, while the S&P was up 1.9% and the Dow was up 1.6%; all three indexes marked their largest reversal from an intraday high to negative territory since April 8, according to Dow Jones Market Data. “This morning’s stunning reversal is a sign that market psychology has be",
    "The Dow Jones Industrial Average rebounded on Friday after New York Federal Reserve President John Williams suggested the central bank could cut interest rates yet again this year. The blue-chip index gained 493.15 points, or 1.08%, to close at 46,245.41. The Nasdaq Composite advanced 0.88% to settle at 22,273.08, while the S&P 500 finished 0.98% higher at 6,602.99. I view monetary policy as being modestly restrictive, although somewhat less so than before our recent actions, Williams said in remarks for a speech in Santiago, Chile. Therefore, I still see room for a further adjustment in the near term to the target range for the federal funds rate to move the stance of policy closer to the range of neutral, thereby maintaining the balance between the achievement of our two goals. The comments by a notable Fed member like Williams signaled to investors that central bank leadership is likely to lower its benchmark overnight borrowing rate at its upcoming December meeting. This led traders to raise bets that the Fed would, in fact, cut next month for the third time in 2025. Fed funds futures are currently pricing in a more than 70% chance of a quarter percentage point cut, a spike from the less than 40% likelihood priced in the day before, according to the CME FedWatch tool. Stocks that could benefit the most from lower rates, which may spur consumer spending, led the market comeback. These included Home Depot, Starbucks and McDonald's. Investors hope easier monetary policy ",
    "Stock market today: Dow, S&P 500, Nasdaq slide out gains as Nvidia, tech stocks lead sharp reversal lower US stocks were hammered on Thursday as an abrupt midday reversal in markets followed a blockbuster earnings report from Nvidia (NVDA) and the release of the long-delayed September jobs report, which showed hiring grew more than expected in the final month of the third quarter. When the closing bell rang on Thursday, the tech-heavy Nasdaq Composite (^IXIC) had taken the sharpest losses, falling more than 2.1% from Wednesday's close, and marking a more than 3.5% intraday reversal. The S&P 500 (^GSPC) fell 1.5%, and the Dow Jones Industrial Average (^DJI) lost 380 points, or 0.8%. Stocks opened the session higher after Nvidia earnings, the September jobs report, and Walmart's strong third quarter lifted investor spirits. The abrupt turn lower in trading on Thursday came alongside a sharp drop in the crypto markets, with bitcoin and ether both falling sharply to add to recent losses. Bitcoin was trading below $87,000 at the close of stock trading on Thursday, near its lowest levels since April. Nvidia stock finished the day down 3.1% after surging as much as 5% in early trading after the chipmaker delivered an earnings beat and issued a stronger-than-expected revenue outlook for the fourth quarter. CEO Jensen Huang said demand for the company’s Blackwell processors is off the charts, helping — at least initially — to ease concern that the recent cooldown in AI-linked stocks",
    "Key Points The Asian social media specialist notched a pair of beats in its third quarter. This, despite declines in revenue and mobile MAUs. NASDAQ: JOYY Joyy The company's recent performance was impressive, but certain financial and operational metrics eroded. Joyy (JOYY 3.00%) is a stock that certainly lived up to its name for investors on Thursday. The Asian social media company's American Depositary Shares (ADSes) rose by 6% in price that trading session, thanks to a pair of beats in its just-reported third quarter. Better-than-expected guidance helped too. Joyy-ful Joyy, which reports in U.S. dollars despite being headquartered in Singapore, published those results early Thursday morning. The company's net revenue was $540 million, down from $559 million in the same period a year ago. Yet it was high enough to top the consensus analyst estimate of $531 million and change. Image source: Getty Images. On the bottom line, net income in accordance with generally accepted accounting practices (GAAP) was 2% higher at $62 million, or $1.15 per ADS. This easily topped the $1.05 collectively expected by the pundits tracking Joyy stock. Although the company suffered a 12% decline in its foundational livestreaming revenue (to under $389 million), this was partially mitigated by a very encouraging 29% rise in advertising revenue -- this came in at almost $113 million. Its monthly average user (MAU) count in the mobile segment decreased to 266 million in the quarter, down from 272 m",
    ]
    summaries = summarize_articles(scraped_articles)

# Print output
    for i, s in enumerate(summaries, 1):
        print(f"\n===== Summary {i} =====")
        print(s)
