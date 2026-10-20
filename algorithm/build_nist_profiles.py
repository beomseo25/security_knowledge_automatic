"""NIST CSF 2.0 용어를 덧붙인 카테고리 설명서(category_profiles_nist.csv)를 만든다.

원본 data/category_profiles_deploy.csv 의 한글 설명은 그대로 두고, 카테고리마다
(1) detail_description 끝에 NIST CSF 2.0 공개 용어로 쓴 영어 설명 한 문장,
(2) keywords 끝에 영어 키워드
를 덧붙인다. 원본 파일은 수정하지 않는다.

작성 원칙 (정직성 기록)
- 근거는 NIST CSF 2.0의 Function/Category 공개 정의 용어와 각 카테고리의 기존 한글 설명이다.
- 1차 결과의 틀린 항목을 보며 고치는 반복은 하지 않았고, NIST ID나 22건의 문장을 그대로 넣지도 않았다.
- 그러나 'CSF의 어떤 Category 용어를 67개 중 어느 카테고리에 붙일지'는 작성자 판단이고, 그 판단은
  정답지와 1차 결과를 이미 본 상태에서 했다. 따라서 이 설명서로 잰 22건 일치율은 독립적인
  값이 아니라 정답지 영향이 섞인 낙관적 값이다. (일반화 성능으로 쓰지 않는다.)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "category_profiles_deploy.csv"
DST = ROOT / "data" / "category_profiles_nist.csv"

# category_id -> (영어 설명 한 문장, 영어 키워드)
NIST_TEXT = {
    "AST-01": ("Inventories of hardware, software, services, systems and data flows are maintained and kept current, with an owner for each asset.",
               "asset inventory, hardware inventory, software inventory, services, systems, data flows, asset owner, maintained"),
    "AST-02": ("Assets are classified and prioritized by criticality, business value and impact if lost.",
               "asset criticality, prioritize assets, classification of assets, resource criticality, business value"),
    "AST-03": ("Rules for acceptable use, return, transfer and secure disposal of assets and storage media.",
               "acceptable use, asset return, media disposal, decommission, sanitize media, remove assets"),
    "AST-04": ("Information is classified, labeled and handled according to its sensitivity.",
               "data classification, labeling, information handling, sensitive information, protect confidentiality"),
    "BCP-01": ("Backups of data are created, protected, maintained and tested so that systems and data can be restored and their integrity verified.",
               "backups, data backup, restore, restoration, verify backup integrity, offline backup, recovery of data"),
    "BCP-02": ("Redundancy, failover and high availability ensure resilience and adequate capacity so that systems keep operating under adverse conditions.",
               "redundancy, failover, high availability, resilience, adequate capacity, avoid single point of failure"),
    "BCP-03": ("Business impact analysis and disaster recovery plans define recovery objectives, and the recovery plan is executed and restoration is prioritized.",
               "business impact analysis, disaster recovery plan, RTO, RPO, recovery objectives, recovery plan execution, restore operations"),
    "BCP-04": ("Business continuity and disaster recovery plans are exercised and tested, and lessons are used to improve them.",
               "business continuity, continuity exercise, disaster recovery test, tabletop, recovery drills, resilience planning"),
    "CLD-01": ("Responsibilities of the cloud service provider and the customer are defined, and cloud service use is governed by agreements.",
               "cloud security, shared responsibility, cloud service provider, cloud governance, cloud agreement"),
    "CLD-02": ("Virtual machines and containers are secured, configured and hardened, and images are managed.",
               "virtualization, containers, virtual machine, image management, hardening, hypervisor"),
    "CLD-03": ("Tenants and cloud resources are isolated from one another.",
               "tenant isolation, multi-tenant, resource isolation, cloud network separation"),
    "CLD-04": ("Cloud-specific protections such as cloud security configuration, key management service and cloud API security.",
               "cloud-specific controls, cloud configuration, cloud API, cloud posture"),
    "CRY-01": ("Data at rest is protected through encryption under a cryptographic policy that defines algorithms and key lengths.",
               "data at rest, encryption, cryptographic policy, algorithm, disk encryption, database encryption, confidentiality of stored data"),
    "CRY-02": ("Data in transit is protected through encrypted channels such as TLS, VPN and SSH.",
               "data in transit, TLS, HTTPS, VPN, SSH, transmission security, encrypted communication, network traffic protection"),
    "CRY-03": ("Cryptographic keys are generated, stored, rotated, backed up and destroyed securely through key management.",
               "cryptographic key management, key lifecycle, key rotation, key storage, key generation, key destruction"),
    "DAT-01": ("Personal and sensitive data is collected, used and retained only as needed, with notice and consent for the stated purpose.",
               "privacy, data collection, data minimization, purpose limitation, consent, notice, personal information use"),
    "DAT-02": ("Provision of data to third parties and outsourcing of personal information processing are controlled.",
               "third party data sharing, data transfer, outsourced processing, personal data provision"),
    "DAT-03": ("Data is retained for defined periods and then securely destroyed or deleted, with destruction verified.",
               "data retention, data deletion, secure destruction, disposal of data, retention period"),
    "DAT-04": ("Data subject rights such as access, rectification, deletion and objection requests are handled.",
               "data subject rights, access request, rectification, deletion request, objection, individual rights"),
    "DAT-05": ("Data is masked, pseudonymized or de-identified when displayed, output or used.",
               "masking, pseudonymization, de-identification, anonymization, display protection"),
    "DAT-06": ("Data leakage is prevented by monitoring and controlling channels through which data leaves the organization.",
               "data leakage prevention, DLP, exfiltration, data loss, removable media control, outbound transfer"),
    "DAT-07": ("Sensitive data such as unique identifiers and health information receives additional protection and impact assessment.",
               "sensitive information, unique identifiers, privacy impact assessment, special categories of data"),
    "DEV-01": ("Security requirements are defined and a secure software development life cycle is applied to systems that are acquired or developed.",
               "secure SDLC, security requirements, secure development, software development life cycle, acquisition"),
    "DEV-02": ("Secure architecture and design principles are established and reviewed for systems.",
               "secure architecture, security design, secure engineering principles, design review"),
    "DEV-03": ("Secure coding rules are followed, and code is reviewed and tested for security flaws before release.",
               "secure coding, code review, security testing, static analysis, vulnerability testing, software quality"),
    "DEV-04": ("Development, test and production environments are separated and transfers between them are controlled.",
               "environment separation, development environment, test environment, production environment, separate environments"),
    "DEV-05": ("Changes to systems, applications and configurations are requested, reviewed, approved, tested, implemented and can be rolled back.",
               "change management, change control, change approval, rollback, configuration change, change request"),
    "DEV-06": ("Outsourced development and test data are handled securely, with security requirements for deliverables.",
               "outsourced development, test data protection, third-party developed software, deliverable acceptance"),
    "GOV-01": ("Cybersecurity policy is established, communicated, enforced, reviewed and updated based on changes in requirements, threats and technology.",
               "cybersecurity policy, policy established, policy communicated, policy enforced, policy review and update, guidelines"),
    "GOV-02": ("The organizational mission and stakeholder expectations are understood and inform cybersecurity risk management; roles, responsibilities and authorities are established, leadership is accountable and resources are allocated.",
               "organizational context, mission, stakeholders, stakeholder expectations, critical objectives, dependencies, roles and responsibilities, authorities, leadership accountability, resources, executive management, CISO"),
    "GOV-03": ("Oversight reviews the results of cybersecurity risk management, measures performance, and continuous improvement acts on the findings of audits, evaluations and lessons learned.",
               "oversight, management review, internal audit, independent review, performance measurement, continuous improvement, evaluations, remedial action, lessons learned, improvement plans"),
    "GOV-04": ("Legal, regulatory and contractual cybersecurity requirements are understood and managed, and operating procedures and records are documented.",
               "legal requirements, regulatory requirements, contractual requirements, compliance, privacy obligations, documented procedures, records management"),
    "HUM-01": ("Personnel are screened before hiring and agree to security responsibilities through confidentiality and employment terms.",
               "personnel screening, background check, hiring, non-disclosure agreement, employment terms"),
    "HUM-02": ("Personnel receive cybersecurity awareness and training suited to their roles, including privileged users and senior executives.",
               "awareness and training, security awareness, role-based training, cybersecurity training, phishing awareness, privileged users trained"),
    "HUM-03": ("Duties are separated and critical roles are protected to reduce misuse; key personnel and administrators are managed.",
               "separation of duties, segregation of duties, key personnel, role separation"),
    "HUM-04": ("Access and assets are revoked at termination or transfer, and security violations are subject to disciplinary action.",
               "termination, personnel transfer, offboarding, revoke access, return of assets, disciplinary process, human resources"),
    "IAM-01": ("Identities are issued, managed, verified, revoked and audited for authorized users, services and hardware throughout their life cycle.",
               "identity management, user accounts, account lifecycle, identities proofed, user registration, unique identification"),
    "IAM-02": ("Users, services and hardware are authenticated; credentials such as passwords and multifactor authentication are managed and commensurate with risk.",
               "authentication, authenticated, credentials, password policy, multifactor authentication, MFA, identity assertions"),
    "IAM-03": ("Access permissions and authorizations follow least privilege and are reviewed and updated.",
               "access permissions, authorizations, least privilege, access review, entitlements, access control policy"),
    "IAM-04": ("Privileged accounts and administrator access are restricted, granted minimally and monitored.",
               "privileged access, administrator accounts, privileged accounts, superuser, privileged access management"),
    "IAM-05": ("Access to information systems and applications is restricted to the functions and data needed.",
               "application access control, system access, access restriction, information access"),
    "IAM-06": ("Remote work, mobile access, and source code access are secured and controlled.",
               "remote access, telework, mobile device access, source code access, remote work security"),
    "INC-01": ("An incident response plan and response organization are established and prepared, with resources, roles and procedures defined before an incident.",
               "incident response plan, incident preparation, response team, response organization, incident handling readiness"),
    "INC-02": ("Incidents and adverse events are detected, reported, triaged, validated and categorized, and criteria are applied to declare an incident and notify stakeholders.",
               "adverse event analysis, anomalies, event assessment, incident report, triage, incident declaration, categorize incidents, prioritize incidents, escalate, notify, incident reporting and communication, share information with stakeholders"),
    "INC-03": ("Incident response is executed: containment, eradication and mitigation limit the spread and effects of an incident, and responders carry out the incident response plan and recovery.",
               "incident management, execute response plan, incident mitigation, contain, containment, eradicate, eradication, isolate, incident analysis, investigation, forensics, root cause, scope and magnitude, recovery actions"),
    "INC-04": ("After an incident, lessons are analyzed and fed back into plans, and response training is conducted.",
               "post-incident review, lessons learned, incident response training, improve response plan, after-action"),
    "LOG-01": ("Logs are generated, collected and kept, protected from tampering, and their integrity is preserved.",
               "audit logs, log management, log collection, log retention, log integrity, logging records"),
    "LOG-02": ("Networks, systems and personnel activity are continuously monitored to find anomalies, indicators of compromise and potentially adverse events.",
               "continuous monitoring, monitor networks, anomalies, indicators of compromise, detection, abnormal activity, security monitoring, event correlation"),
    "LOG-03": ("System clocks are synchronized to a standard time source so that events can be correlated.",
               "time synchronization, NTP, clock synchronization, timestamps"),
    "NET-01": ("Network access is controlled, with rules for which devices and users can connect, and networks are protected against unauthorized access.",
               "network access control, network protection, network security, unauthorized connection, device authentication"),
    "NET-02": ("Network security devices such as firewalls and the services running on the network are configured and managed with a policy.",
               "firewall, network services, network device policy, rule management, boundary protection"),
    "NET-03": ("Network zones are segmented and isolated by sensitivity, and communication between segments is limited.",
               "network segmentation, zones, DMZ, isolation, separate networks, network integrity protected"),
    "NET-04": ("Internet-facing services and traffic are protected with controls such as web application firewall and filtering of malicious sites.",
               "internet-facing, web application firewall, WAF, web filtering, email security, external connection"),
    "OPS-01": ("Vulnerabilities in assets are identified, assessed and remediated, and patches are applied based on risk.",
               "vulnerability management, vulnerability scanning, patching, vulnerabilities identified, remediation of vulnerabilities, software maintenance, security updates"),
    "OPS-02": ("Malicious code is detected and prevented, and endpoints are protected with anti-malware.",
               "malware, malicious code, anti-malware, antivirus, endpoint protection, endpoint detection"),
    "OPS-03": ("Capacity of system resources is monitored and planned so that services do not fail from shortage.",
               "capacity management, resource usage, capacity planning, performance monitoring, availability"),
    "OPS-04": ("Operational software and programs are controlled; installation and changes to production systems are managed, and secure configuration baselines are kept.",
               "configuration management, baseline configuration, software installation control, authorized software, hardening, platform security"),
    "OPS-05": ("Servers, databases and applications are securely configured and hardened, and security settings are maintained.",
               "server hardening, database security, secure configuration, operating system security, application security settings"),
    "PHY-01": ("Physical access to facilities and critical areas is restricted and managed by physical perimeters and access records.",
               "physical access control, secured areas, restricted area, badge, visitor access, physical security perimeter"),
    "PHY-02": ("Facilities are protected from environmental threats such as fire, flood and power failure.",
               "environmental threats, fire protection, power failure, facility protection, temperature humidity, data center environment"),
    "PHY-03": ("The physical environment is monitored for unauthorized access and abnormal events.",
               "physical monitoring, CCTV, surveillance cameras, intrusion alarm, monitor physical environment"),
    "PHY-04": ("Equipment, external assets and portable devices are protected, including placement and removal.",
               "equipment protection, device security, off-site equipment, portable media, external assets"),
    "RSK-01": ("Risks are identified, assessed and analyzed: threats, vulnerabilities, likelihood and impact are determined, and the risk assessment method is standardized.",
               "risk assessment, risk identification, threats, likelihood, impact, risk analysis, risk management objectives, risk appetite, risk tolerance, risk management strategy, risk register"),
    "RSK-02": ("Risks are treated, responded to and tracked; threat intelligence informs risk decisions, and risk responses are chosen, prioritized and monitored.",
               "risk treatment, risk response, risk acceptance, residual risk, threat intelligence, risk communication, risk mitigation plan"),
    "SUP-01": ("Suppliers and third parties are identified, risk-assessed and selected, and cybersecurity requirements are established in contracts and agreements.",
               "supply chain risk management, suppliers, third parties, supplier selection, contract requirements, supply chain, vendor due diligence, supply chain risk strategy, cybersecurity roles for suppliers"),
    "SUP-02": ("Risks across the supply chain of products and services are understood and managed, including outsourced services and the supplier relationship.",
               "outsourcing, supplier risk, supply chain, subcontractors, product and service supply chain"),
    "SUP-03": ("Supplier performance and security are monitored and reviewed, and relationships end with return or destruction of data.",
               "monitor suppliers, supplier assessment, service delivery review, supplier audit, termination of supplier, return of data"),
}


def main() -> int:
    if DST.exists():
        print("[안내] %s 가 이미 있습니다. 덮어쓰지 않습니다." % DST)
        return 1
    with open(SRC, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    missing = [r["category_id"] for r in rows if r["category_id"] not in NIST_TEXT]
    extra = [k for k in NIST_TEXT if k not in {r["category_id"] for r in rows}]
    if missing or extra:
        print("[오류] 누락 %s / 불필요 %s" % (missing, extra))
        return 1
    for r in rows:
        sentence, keywords = NIST_TEXT[r["category_id"]]
        r["detail_description"] = r["detail_description"].rstrip() + " " + sentence
        r["keywords"] = r["keywords"].rstrip().rstrip(",") + ", " + keywords
    with open(DST, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("저장: %s (%d개)" % (DST, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
