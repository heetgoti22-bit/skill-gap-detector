"""Test suite. Run: pytest tests/ -v"""
import pytest,tempfile,asyncio,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
from pipeline import Database,SkillExtractor,TrendAnalyzer,LearningPathMapper,RateLimiter,Ingester,ALIAS_MAP,SKILL_INDUSTRY,TAXONOMY,create_api
from unittest.mock import patch
import time

@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as td:yield Database(f"{td}/test.db")

class TestDatabase:
    def test_insert(s,tmp_db):assert tmp_db.insert_posting({"id":"t1","title":"Eng","description":"x","company":"Co","source":"test"})==True
    def test_id_dedup(s,tmp_db):
        tmp_db.insert_posting({"id":"t1","title":"Eng","description":"x","company":"Co","source":"test"})
        assert tmp_db.insert_posting({"id":"t1","title":"Eng","description":"y","company":"Co","source":"test"})==False
    def test_content_dedup(s,tmp_db):
        tmp_db.insert_posting({"id":"a","title":"ML Eng","description":"x","company":"Co","source":"gh"})
        assert tmp_db.insert_posting({"id":"b","title":"ML Eng","description":"y","company":"Co","source":"lv"})==False
    def test_different_inserts(s,tmp_db):
        tmp_db.insert_posting({"id":"a","title":"ML","description":"x","company":"A","source":"t"})
        tmp_db.insert_posting({"id":"b","title":"Data","description":"y","company":"B","source":"t"})
        assert tmp_db.stats()["postings"]==2
    def test_skills(s,tmp_db):
        tmp_db.insert_posting({"id":"s1","title":"E","description":"x","company":"C","source":"t","posted_date":"2025-01-15"})
        tmp_db.insert_posting({"id":"s2","title":"E2","description":"y","company":"D","source":"t","posted_date":"2025-02-15"})
        tmp_db.insert_skills("s1",[{"skill":"python","method":"taxonomy","confidence":1.0}])
        assert len(tmp_db.get_postings_without_skills())==1
    def test_schema_version(s,tmp_db):
        with tmp_db.conn() as c:assert c.execute("SELECT version FROM schema_version").fetchone()[0]==Database.CURRENT_VERSION
    def test_trends(s,tmp_db):
        tmp_db.save_trends([{"skill":"python","direction":"rising","growth_yoy":25,"current_demand":5000,"avg_salary":150000,"confidence":0.87,"forecast":[],"industry":"Languages","tags":["python3"],"monthly_counts":{}}])
        t=tmp_db.get_trends(direction="rising");assert len(t)==1;assert t[0]["skill"]=="python"

class TestTaxonomy:
    def test_aliases(s):
        assert ALIAS_MAP["k8s"]=="kubernetes";assert ALIAS_MAP["langchain"]=="ai agent frameworks"
        assert ALIAS_MAP["ml ops"]=="mlops";assert ALIAS_MAP["rlhf"]=="llm fine-tuning"
        assert ALIAS_MAP["sbom"]=="supply chain security";assert ALIAS_MAP["argocd"]=="gitops"
    def test_industry(s):assert SKILL_INDUSTRY["mlops"]=="AI/ML";assert SKILL_INDUSTRY["kubernetes"]=="Cloud"

class TestExtraction:
    def test_patterns(s,tmp_db):
        ext=SkillExtractor(tmp_db);text="kubernetes, docker, and machine learning";found=set()
        for pat,c in ext._patterns:
            if pat.search(text):found.add(c)
        assert "kubernetes" in found;assert "docker" in found;assert "machine learning" in found
    def test_requirements(s,tmp_db):
        ext=SkillExtractor(tmp_db);text="About us: Great.\nRequirements:\n- Python\n- Kubernetes\nNice to have:\n- Rust"
        req=ext._extract_req(ext._preprocess(text));assert "Python" in req
    def test_pattern_sort(s,tmp_db):
        ext=SkillExtractor(tmp_db);lengths=[len(p[0].pattern) for p in ext._patterns]
        assert lengths==sorted(lengths,reverse=True)

class TestTrends:
    def test_mk_rising(s,tmp_db):
        a=TrendAnalyzer(tmp_db);mk=a._mann_kendall([10,15,20,25,30,35,40,45,50,55,60,65])
        assert mk["trend_detected"];assert mk["p_value"]<0.05
    def test_mk_falling(s,tmp_db):
        a=TrendAnalyzer(tmp_db);assert a._mann_kendall([65,60,55,50,45,40,35,30,25,20,15,10])["trend_detected"]
    def test_mk_flat(s,tmp_db):assert not TrendAnalyzer(tmp_db)._mann_kendall([50,48,52,49,51,50,48,52,49,51,50,48])["trend_detected"]
    def test_sens_slope(s,tmp_db):
        sl=TrendAnalyzer(tmp_db)._sens_slope([10,15,20,25,30,35,40,45,50,55,60,65]);assert 4<sl<6
    def test_classify_both(s,tmp_db):
        a=TrendAnalyzer(tmp_db)
        assert a._classify(0.5,{"trend_detected":False,"p_value":0.3,"statistic":5},[])=="stable"
        assert a._classify(0.5,{"trend_detected":True,"p_value":0.001,"statistic":20},[])=="rising"
    def test_forecast(s,tmp_db):
        a=TrendAnalyzer(tmp_db);m=[f"2025-{i:02d}" for i in range(1,13)];v=[100+i*10 for i in range(12)]
        fc=a._linear_forecast(m,v,3);assert len(fc)==3;assert fc[0]["predicted"]>v[-1]

class TestSalary:
    def test_range(s,tmp_db):
        ing=Ingester.__new__(Ingester);assert ing._parse_salary("$120,000 - $180,000")==(120000,180000)
    def test_k(s,tmp_db):
        ing=Ingester.__new__(Ingester);assert ing._parse_salary("$120k-$180k")==(120000,180000)
    def test_none(s,tmp_db):
        ing=Ingester.__new__(Ingester);assert ing._parse_salary("competitive")==(None,None)
    def test_reject_low(s,tmp_db):
        ing=Ingester.__new__(Ingester);assert ing._parse_salary("$5 - $10")==(None,None)

class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_burst(s):
        rl=RateLimiter(10.0);t0=time.monotonic()
        for _ in range(5):await rl.acquire()
        assert time.monotonic()-t0<2.0

class TestAPI:
    @pytest.fixture
    def client(s,tmp_db):
        tmp_db.save_trends([
            {"skill":"python","direction":"rising","growth_yoy":25,"current_demand":10000,"avg_salary":150000,"confidence":0.9,"forecast":[],"industry":"Languages","tags":[],"monthly_counts":{}},
            {"skill":"hadoop","direction":"declining","growth_yoy":-35,"current_demand":500,"avg_salary":120000,"confidence":0.8,"forecast":[],"industry":"Legacy","tags":[],"monthly_counts":{}},
        ])
        with patch("pipeline.Database",return_value=tmp_db):app=create_api()
        from starlette.testclient import TestClient;return TestClient(app)
    def test_health(s,client):r=client.get("/api/v1/health");assert r.status_code==200
    def test_trends(s,client):r=client.get("/api/v1/trends");assert r.status_code==200;assert r.json()["count"]==2
    def test_trends_filter(s,client):r=client.get("/api/v1/trends?direction=rising");assert all(t["direction"]=="rising" for t in r.json()["trends"])
    def test_skill_found(s,client):r=client.get("/api/v1/skills/python");assert r.status_code==200
    def test_skill_404(s,client):assert client.get("/api/v1/skills/nonexistent").status_code==404
    def test_gap(s,client):
        r=client.post("/api/v1/gap-analysis",json={"skills":["python"]});assert r.status_code==200;assert "readiness_score" in r.json()
    def test_gap_empty(s,client):assert client.post("/api/v1/gap-analysis",json={"skills":[]}).status_code==422
    def test_methodology(s,client):r=client.get("/api/v1/methodology");assert r.status_code==200;assert "known_limitations" in r.json()

class TestLearningPaths:
    def test_prereqs_valid(s):
        m=LearningPathMapper.__new__(LearningPathMapper)
        for skill,prereqs in m.PREREQS.items():
            assert skill in TAXONOMY,f"{skill} not in taxonomy"
            for p in prereqs:assert p in TAXONOMY,f"prereq {p} not in taxonomy"
    def test_career_skills_valid(s):
        m=LearningPathMapper.__new__(LearningPathMapper)
        for role,path in m.CAREER_PATHS.items():
            for sk in path["skills"]:assert sk in TAXONOMY,f"{role}: {sk} not in taxonomy"
