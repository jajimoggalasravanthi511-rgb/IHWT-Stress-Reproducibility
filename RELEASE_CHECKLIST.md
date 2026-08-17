# GitHub and DOI Release Checklist

- [ ] Verify the native ECG/GSR/PPG sampling frequencies against acquisition records.
- [ ] Verify actual sensor models, resolution, placement, calibration, and filter settings.
- [ ] Replace every placeholder in the manuscript, including motion-artifact discard percentage.
- [ ] Place de-identified custom data in a shareable repository only if ethics/consent allows it.
- [ ] Run `pytest -q`.
- [ ] Run the complete LOSO experiment on the custom dataset.
- [ ] Run the public WESAD external benchmark.
- [ ] Run WPT and modern deep baselines.
- [ ] Confirm all numerical manuscript values against generated CSV outputs.
- [ ] Regenerate figures at publication quality; replace the cropped Figure 1.
- [ ] Commit the final `config.yaml`, code, and documentation to GitHub.
- [ ] Create a versioned GitHub release.
- [ ] Archive that exact release in Zenodo or another recognized DOI-assigning repository.
- [ ] Record the issued DOI in `CODE_AVAILABILITY.md`, `CITATION.cff`, README, and the manuscript Code Availability section.
- [ ] If a separate data DOI is issued, record it in `DATA_AVAILABILITY.md` and the manuscript Data Availability section.
- [ ] Re-run the final command from a clean environment and retain `pip freeze` and hardware details.
