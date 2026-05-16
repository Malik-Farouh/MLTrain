using namespace TMVA::Experimental;

void CreateInference(){
    string modelname = "model/TrainBkgDiag.h5";
    string infername = "model/TrainBkgDiag.hxx";
    
    std::cout << "Attempting to parse: " << modelname << std::endl;

    // In 6.30, we use the PyKeras sub-namespace directly
    auto model = SOFIE::PyKeras::Parse(modelname);

    // Generating inference code
    model.Generate();
    
    std::cout << "Saving file " << infername << std::endl;
    model.OutputGenerated(infername);
}
