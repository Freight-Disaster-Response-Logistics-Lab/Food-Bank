# A Location-Allocation Model for Food Distribution in Post-Disaster Environments

This repository contains the code and data associated with the research paper **"A Location-Allocation Model for Food Distribution in Post-Disaster Environments"**, authored by Diana Ramirez-Rios, Angelo Soto-Vergel, Trilce Encarnacion, and Johanna Amaya.

## 📖 Abstract

This research investigates the optimal location decisions in a food distribution network during post-disaster environments. We propose a location-allocation model for a food bank network that minimizes the social costs of human suffering caused by delays in distributing food supplies to disaster survivors. The model incorporates an empirically estimated deprivation cost function for food and water supply, which is non-linear with respect to the survivor’s deprivation time. We tested the model using the Houston Food Bank network, aiming to simulate relief efforts following Hurricane Harvey in 2017.

## 📌 Key Features

- **Optimization Model**: Mixed-integer programming (MIP) model for facility location-allocation.
- **Deprivation Cost Function**: First application of an empirically estimated food deprivation cost function in a food bank setting.
- **Case Study**: Application of the model to the Houston Food Bank network during a post-disaster scenario.
- **Social Costs Minimization**: Incorporates logistics and deprivation costs to prioritize survivors’ needs.

## 🗂 Repository Structure

- `data/` - Contains input datasets, including food bank locations, demand zones, and transportation costs.
- `models/` - Python scripts for running the optimization model.
- `results/` - Output results, including optimal POD locations and shipment schedules.
- `notebooks/` - Jupyter notebooks for data preprocessing and analysis.
- `docs/` - Additional documentation and references.

## 🚀 Installation & Usage

### Prerequisites

- Python 3.x
- Gurobi Optimizer
- Required Python Libraries:
  ```bash
  pip install gurobipy pandas numpy matplotlib
  ```

### Running the Model

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Food_Bank_POD_Planning.git
   cd Food_Bank_POD_Planning
   ```

## 📊 Results & Insights

The model provides optimal locations for Points of Distribution (PODs) and delivery strategies to minimize human suffering and logistical costs. The case study on the Houston Food Bank demonstrated that:

- A **network of PODs can be activated** efficiently to serve the impacted population.
- **Social costs-based optimization** ensures equitable distribution of resources.
- **Larger but less frequent shipments** are better for remote PODs, whereas **smaller but frequent shipments** benefit PODs near food banks.
- **Volunteer efforts can significantly reduce deprivation costs** by expediting unloading and distribution.

## 📌 Citation

If you use this work, please cite:

```bibtex
@article{RamirezRios2025FoodBank,
  author    = {Diana Ramirez-Rios and Angelo Soto-Vergel and Trilce Encarnacion and Johanna Amaya},
  title     = {A Location-Allocation Model for Food Distribution in Post-Disaster Environments},
  journal   = {Networks and Spatial Economics},
  DOI       = 10.1007/s11067-025-09681-3,
  year      = {2025}
}
```

## 📜 License

This project is licensed under the MIT License.

## 📬 Contact

For questions or collaborations, please contact:

- **Diana Ramirez-Rios** - dgramire@buffalo.edu
- **Angelo Soto-Vergel** - angeloso@buffalo.edu
- **Trilce Encarnacion** - tencarnacion@umsl.edu
- **Johanna Amaya** - amayaj@psu.edu

---

Made with ❤️ for humanitarian logistics & disaster relief research! 🚀
