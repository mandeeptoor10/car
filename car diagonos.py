def diagnose(session):
    dtcs = session.dtcs
    telemetry = session.telemetry_window()  # last 60s
    findings = []

    # 1) Rule-based mapping
    for code in dtcs:
        mapping = DTC_DB.lookup(code)
        findings.append({"code": code, "mapping": mapping})

    # 2) Run symptom checks
    if "P0302" in dtcs:
        ft = telemetry.mean("short_term_fuel_trim_bank1")
        o2 = telemetry.mean("o2_voltage_bank1")
        if ft > 15 and o2 < 0.2:
            findings.append({"probable": "injector_cylinder_2", "reason":"high fuel trim, low O2"})

    # 3) ML model (returns ranked causes)
    features = build_features(telemetry, dtcs, session.meta)
    ml_ranking = ml_model.predict_rank(features)
    findings += [{"ml_ranked": ml_ranking}]

    # 4) Compose final report with confidence
    report = aggregate_findings(findings)
    return report