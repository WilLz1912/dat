"""Từ khoá nhận khía cạnh (mốc so sánh). Viết sau khi đã đọc câu test nên điểm có thể hơi lạc quan."""
import re

KEYWORDS = {
    "GFX": r"graphic|visual|art ?style|\bart\b|textures?\b|animation|beautiful|gorgeous|\bugly\b|aesthetic|looks? (great|good|nice|bad|amazing|awful)",
    "AUD": r"\bsounds?\b|\baudio\b|\bmusic\b|soundtrack|\bost\b|voice ?act|\bvoices?\b|voicework",
    "PLAY": r"gameplay|mechanic|\bcombat\b|\bcontrols?\b|\bgameloop\b|\bloop\b|\bgrind|repetitive|difficult|\bbalanc|\bui\b|\bboss(es)?\b|\bbuilding\b|\bexplor|\bprogression\b|\bmovement\b|\bcamera\b",
    "STORY": r"\bstor(y|ies|yline)\b|\bplot\b|narrative|\bcharacters?\b|\bwriting\b|\bending\b|\bquests?\b|\blore\b|dialogue",
    "PERF": r"\bbugs?\b|buggy|\bcrash|\blag|\bfps\b|\bframes?\b|optimi[sz]|stutter|\bfreez|\bbroken\b|performance|\bglitch|unplayable|won'?t (launch|start)|\brun(s|ning)? (well|bad|badly|fine|poorly)",
    "PRICE": r"\bprice|\bmoney\b|\brefund|\bworth\b|overpriced|\bcheap\b|\bdlcs?\b|microtransaction|macrotransaction|\bmtx\b|pay to win|\bp2w\b|\bcost|\bpaid\b|\bsale\b|battle ?pass",
    "ONLINE": r"multiplayer|\bservers?\b|matchmaking|match making|\bonline\b|\bco-?op\b|\bfriends?\b|\bpvp\b|\bcheat|\bhacker|\bhacks?\b|anti-?cheat|\bcommunity\b|toxic|\bteammates?\b|\bsquads?\b|\bplayers\b|\blobb(y|ies)\b",
    "DEV": r"\bdevs?\b|developers?\b|\bpublisher\b|\bupdates?\b|\bpatch(es)?\b|early access|abandon|\bsupport\b|\bvalve\b|\bea\b|\bubisoft\b|\bblizzard\b|\brockstar\b|\bbethesda\b|\bparadox\b",
}
KW_RE = {k: re.compile(v, re.I) for k, v in KEYWORDS.items()}


def detect(t):
    return {k for k, r in KW_RE.items() if r.search(t)}
