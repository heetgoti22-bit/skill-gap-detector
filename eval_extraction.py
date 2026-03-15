"""Extraction quality evaluation against 50 hand-labeled postings."""
import json,sys,re
from pathlib import Path
from datetime import datetime
from collections import Counter,defaultdict

LABELED=[
{"id":"e01","text":"Senior ML Engineer. Python, PyTorch, MLflow-based MLOps platform. Kubernetes, Docker. LLM fine-tuning (LoRA, QLoRA) strong plus.","gt":["machine learning","python","mlops","kubernetes","docker","llm fine-tuning"]},
{"id":"e02","text":"Data Engineer. SQL and Python. dbt, Apache Spark, Snowflake. Apache Kafka preferred. Terraform for infrastructure.","gt":["python","dbt","apache spark","snowflake","apache kafka","terraform"]},
{"id":"e03","text":"Security Engineer. 3+ years cloud security (AWS, GCP, Azure). Zero trust architecture and IAM. DevSecOps and CI/CD pipeline security.","gt":["cloud security","aws","gcp","azure","zero trust","identity management","devsecops","ci/cd"]},
{"id":"e04","text":"Frontend Engineer. React with TypeScript and Next.js. Tailwind CSS preferred. Design system components.","gt":["react","typescript","next.js"]},
{"id":"e05","text":"AI Research Scientist. AI safety and alignment. PhD in ML. RLHF and LLM evaluation. Transformer architectures.","gt":["ai safety","machine learning","llm fine-tuning","transformers"]},
{"id":"e06","text":"Platform Engineer. Kubernetes expertise. Backstage, Terraform, GitOps (ArgoCD/Flux). CI/CD. Observability (OpenTelemetry, Grafana).","gt":["platform engineering","kubernetes","terraform","gitops","ci/cd","observability"]},
{"id":"e07","text":"Senior Software Engineer - Rust. WebAssembly preferred. C++ helpful.","gt":["rust","webassembly","c++"]},
{"id":"e08","text":"Product Manager - AI. Prompt engineering, A/B testing. LLM capabilities and RAG architectures.","gt":["prompt engineering","rag systems"]},
{"id":"e09","text":"DevOps. Docker, Kubernetes, AWS. CI/CD (GitHub Actions/Jenkins). Chaos engineering (LitmusChaos). Prometheus and Grafana.","gt":["docker","kubernetes","aws","ci/cd","chaos engineering","observability"]},
{"id":"e10","text":"ML Engineer - NLP. NLP pipelines, transformers, Hugging Face, Python. Deploy with MLflow and Docker.","gt":["natural language processing","transformers","python","mlops","docker"]},
{"id":"e11","text":"Data Scientist. Python, SQL. RAG systems with vector databases (Pinecone, Weaviate). LangChain preferred.","gt":["python","rag systems","vector databases","ai agent frameworks"]},
{"id":"e12","text":"SRE. Kubernetes, Terraform IaC, observability (OpenTelemetry). Service mesh (Istio/Linkerd).","gt":["site reliability","kubernetes","terraform","observability","service mesh"]},
{"id":"e13","text":"Full Stack. React, Node.js, TypeScript. AWS, Docker.","gt":["react","typescript","aws","docker"]},
{"id":"e14","text":"Edge AI. TensorFlow Lite. Rust and C++ for firmware.","gt":["edge ai","rust","c++"]},
{"id":"e15","text":"Security Ops. Penetration testing. Cloud security (AWS).","gt":["penetration testing","cloud security","aws"]},
{"id":"e16","text":"Data Platform. Kafka streaming, Spark batch. dbt. Databricks with Delta Lake. Data governance.","gt":["apache kafka","apache spark","dbt","databricks","data lakehouse","data governance"]},
{"id":"e17","text":"AI Red Team. Test LLM vulnerabilities. Prompt injection, adversarial ML. Python. Penetration testing helpful.","gt":["ai red teaming","python","penetration testing"]},
{"id":"e18","text":"ML Infra. GPU orchestration for training. Kubernetes, Ray, NVIDIA Triton. Docker, Terraform.","gt":["gpu orchestration","kubernetes","docker","terraform"]},
{"id":"e19","text":"Backend. Go (Golang). Kubernetes microservices. CI/CD with GitHub Actions.","gt":["go","kubernetes","ci/cd"]},
{"id":"e20","text":"FinOps. Cloud cost optimization AWS and GCP. Kubernetes resource management. Grafana dashboards.","gt":["finops","aws","gcp","kubernetes","observability"]},
{"id":"e21","text":"Gen AI Engineer. LLMs, RAG, vector databases, prompt engineering. Python, LangChain. Fine-tuning (LoRA).","gt":["generative ai","rag systems","vector databases","prompt engineering","python","ai agent frameworks","llm fine-tuning"]},
{"id":"e22","text":"Supply chain security. SBOM generation, CI/CD. DevSecOps. Container scanning. Zero trust.","gt":["supply chain security","ci/cd","devsecops","zero trust"]},
{"id":"e23","text":"Analytics Engineer. dbt and SQL. Snowflake or BigQuery. Python.","gt":["dbt","snowflake","gcp","python"]},
{"id":"e24","text":"Chaos Engineer. Failure injection. Kubernetes. Observability (OpenTelemetry). SRE background.","gt":["chaos engineering","kubernetes","observability","site reliability"]},
{"id":"e25","text":"AIOps Engineer. ML-powered monitoring. Python. Prometheus, Grafana, OpenTelemetry.","gt":["aiops","python","observability"]},
{"id":"e26","text":"Computer Vision. Real-time object detection. TensorFlow Lite, ONNX. Python. Docker.","gt":["computer vision","edge ai","python","docker"]},
{"id":"e27","text":"Java Engineer. Spring Boot microservices. AWS (ECS, RDS). CI/CD Jenkins. Docker. Kotlin migration.","gt":["java","aws","ci/cd","docker"]},
{"id":"e28","text":"Feature Store Engineer. Feast or Tecton. Apache Spark, Kafka. Python.","gt":["feature stores","apache spark","apache kafka","python"]},
{"id":"e29","text":"Cloud Architect GCP. Kubernetes (GKE), Terraform, serverless (Cloud Functions). Security.","gt":["gcp","kubernetes","terraform","serverless","cloud security"]},
{"id":"e30","text":"Data Mesh Architect. Domain-driven data. Data governance, Spark, dbt. Self-serve platform.","gt":["data mesh","data governance","apache spark","dbt"]},
{"id":"e31","text":"Synthetic Data Engineer. Generative models. Python, PyTorch. Diffusion models, GANs.","gt":["synthetic data","python","generative ai","deep learning"]},
{"id":"e32","text":"Developer Portal Lead. Backstage. Kubernetes. API design.","gt":["platform engineering","kubernetes"]},
{"id":"e33","text":"Streaming Engineer. Apache Flink. Kafka. Python and Java. Kubernetes. Prometheus.","gt":["apache flink","apache kafka","python","java","kubernetes","observability"]},
{"id":"e34","text":"MLOps Engineer. Kubeflow, Airflow. Docker, Kubernetes. MLflow. Python. CI/CD for ML.","gt":["mlops","apache airflow","docker","kubernetes","python","ci/cd"]},
{"id":"e35","text":"Pen Tester. Web app security, OWASP. Python. AWS, Azure.","gt":["penetration testing","python","aws","azure"]},
{"id":"e36","text":"GitOps Engineer. ArgoCD deployments. Kubernetes. Terraform. OPA/Kyverno policy.","gt":["gitops","kubernetes","terraform"]},
{"id":"e37","text":"Deep Learning Researcher. Novel architectures. Multi-modal models. GPU optimization.","gt":["deep learning","multimodal ml","gpu orchestration"]},
{"id":"e38","text":"IAM Engineer. SSO, OAuth 2.0, OIDC. Passkeys/FIDO2. AWS IAM. Zero trust.","gt":["identity management","aws","zero trust"]},
{"id":"e39","text":"Maintain legacy PHP, migrate to TypeScript/React. jQuery cleanup.","gt":["typescript","react"]},
{"id":"e40","text":"Vector DB Engineer. Pinecone/Weaviate. Embedding pipelines. Python, Docker. RAG architecture.","gt":["vector databases","python","docker","rag systems"]},
{"id":"e41","text":"Airflow Developer. Data pipelines. Python, SQL. dbt. AWS (S3, Redshift).","gt":["apache airflow","python","dbt","aws"]},
{"id":"e42","text":"Prompt Engineer. Design prompts for production LLMs. A/B test variants. Python.","gt":["prompt engineering","python"]},
{"id":"e43","text":"Graph Database Engineer. Neo4j. Knowledge graphs for RAG. Python.","gt":["python","rag systems"]},
{"id":"e44","text":"Cloud FinOps. Reduce AWS and GCP spend. Kubernetes resource optimization.","gt":["finops","aws","gcp","kubernetes"]},
{"id":"e45","text":"Multimodal ML. Vision-language models. CLIP, LLaVA. GPU training at scale.","gt":["multimodal ml","deep learning","gpu orchestration"]},
{"id":"e46","text":"Serverless. AWS Lambda, API Gateway. Terraform. Python and TypeScript.","gt":["serverless","aws","terraform","python","typescript"]},
{"id":"e47","text":"Hadoop/MapReduce migration to Spark and Delta Lake. Python. Airflow orchestration.","gt":["hadoop","apache spark","data lakehouse","python","apache airflow"]},
{"id":"e48","text":"AI Agent Dev. LangChain, CrewAI. RAG with Pinecone. Python. Prompt engineering.","gt":["ai agent frameworks","rag systems","vector databases","python","prompt engineering"]},
{"id":"e49","text":"Edge Computing. ML on edge devices. Rust for performance. Docker for ARM.","gt":["edge ai","rust","docker"]},
{"id":"e50","text":"Observability Engineer. OpenTelemetry. Grafana. Prometheus. Kubernetes. Python.","gt":["observability","kubernetes","python"]},
]

def run():
    sys.path.insert(0,str(Path(__file__).parent))
    from pipeline import SkillExtractor,Database,ALIAS_MAP
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        db=Database(f"{td}/eval.db");ext=SkillExtractor(db)
    all_tp=all_fp=all_fn=0;results=[]
    for posting in LABELED:
        text=ext._preprocess(posting["text"]);req=ext._extract_req(text);target=req if len(req)>100 else text
        extracted=set();seen=set()
        for pat,canonical in ext._patterns:
            if canonical in seen:continue
            if pat.search(target):extracted.add(canonical);seen.add(canonical)
        try:
            if len(target)<8000:
                doc=ext.nlp(target[:5000])
                for ent in doc.ents:
                    if ent.label_=="TECH_SKILL":
                        c=ALIAS_MAP.get(ent.text.lower())
                        if c and c not in seen:extracted.add(c);seen.add(c)
        except:pass
        gt=set(posting["gt"]);tp=extracted&gt;fp=extracted-gt;fn=gt-extracted
        all_tp+=len(tp);all_fp+=len(fp);all_fn+=len(fn)
        results.append({"id":posting["id"],"gt":sorted(gt),"extracted":sorted(extracted),"tp":sorted(tp),"fp":sorted(fp),"fn":sorted(fn)})
    precision=all_tp/max(1,all_tp+all_fp);recall=all_tp/max(1,all_tp+all_fn);f1=2*precision*recall/max(0.001,precision+recall)
    fn_counter=Counter();fp_counter=Counter()
    for r in results:
        for s in r["fn"]:fn_counter[s]+=1
        for s in r["fp"]:fp_counter[s]+=1
    print(f"\n{'='*50}");print(f"EXTRACTION EVALUATION — {len(LABELED)} postings");print(f"{'='*50}")
    print(f"\n  Precision:  {precision:.1%}");print(f"  Recall:     {recall:.1%}");print(f"  F1 Score:   {f1:.1%}")
    print(f"  TP: {all_tp}  FP: {all_fp}  FN: {all_fn}")
    if fn_counter:print(f"\n  Most common misses:");
    for s,c in fn_counter.most_common(5):print(f"    {s}: missed {c}x")
    if fp_counter:print(f"\n  Most common false positives:");
    for s,c in fp_counter.most_common(5):print(f"    {s}: wrong {c}x")
    print(f"\n  Errors (first 3):")
    ec=0
    for r in results:
        if r["fp"] or r["fn"]:
            ec+=1
            if ec>3:break
            print(f"    {r['id']}: expected={r['gt']}")
            if r["fp"]:print(f"      FP: {r['fp']}")
            if r["fn"]:print(f"      FN: {r['fn']}")
    Path("data/eval").mkdir(parents=True,exist_ok=True)
    with open("data/eval/eval_results.json","w") as f:json.dump({"date":datetime.now().isoformat(),"n":len(LABELED),"precision":round(precision,3),"recall":round(recall,3),"f1":round(f1,3),"results":results},f,indent=2)
    print(f"\n  Results saved to data/eval/eval_results.json")
    md=f"# Extraction Eval — {datetime.now().strftime('%Y-%m-%d')}\n\n| Metric | Value |\n|--------|-------|\n| Precision | {precision:.1%} |\n| Recall | {recall:.1%} |\n| F1 | {f1:.1%} |\n| Postings | {len(LABELED)} |\n"
    with open("data/eval/eval_report.md","w") as f:f.write(md)
    print(f"  Report saved to data/eval/eval_report.md\n")

if __name__=="__main__":run()
