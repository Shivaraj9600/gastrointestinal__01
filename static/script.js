document.addEventListener('DOMContentLoaded', () => {
    const downloadPdfBtn = document.getElementById('download-pdf-btn');

    if (downloadPdfBtn) {
        downloadPdfBtn.addEventListener('click', () => {
            
            // Visual feedback
            const originalBtnText = downloadPdfBtn.innerHTML;
            downloadPdfBtn.innerHTML = "⏳ Generating PDF...";
            downloadPdfBtn.disabled = true;

            // Short delay to let the button UI update before processing
            setTimeout(() => {
                try {
                    // Initialize the pure jsPDF library
                    const { jsPDF } = window.jspdf;
                    const doc = new jsPDF('p', 'mm', 'a4'); // Portrait, Millimeters, A4 size

                    // 1. Write the Header Text
                    doc.setFontSize(22);
                    doc.text("Gastrointestinal Diagnosis Report", 14, 20);

                    doc.setFontSize(12);
                    doc.setTextColor(100); // Gray color
                    doc.text(`Model Used: ${document.getElementById('model-used-display').innerText}`, 14, 30);
                    
                    doc.setTextColor(0); // Black color
                    doc.text(`Prediction: ${document.getElementById('result-prediction').innerText}`, 14, 40);
                    doc.text(`Confidence: ${document.getElementById('result-confidence').innerText}`, 14, 48);

                    // 2. Grab your high-res images
                    const imgs = window.pdfImages || {};
                    const pageWidth = doc.internal.pageSize.getWidth();
                    const margin = 14;
                    const imgWidth = pageWidth - (margin * 2);
                    const imgHeight = (imgWidth * 0.75); // Standard 4:3 aspect ratio

                    let yOffset = 65;

                    // 3. Inject Grad-CAM (Page 1)
                    if (imgs.gradcam) {
                        doc.setFontSize(14);
                        doc.text("1. Grad-CAM Analysis", margin, yOffset);
                        // Inject directly. No canvas rendering = no memory crash!
                        doc.addImage(imgs.gradcam, 'PNG', margin, yOffset + 5, imgWidth, imgHeight); 
                    }

                    // 4. Inject LIME (Page 2)
                    if (imgs.lime) {
                        doc.addPage(); // Put it on a fresh page so it looks professional
                        yOffset = 20;
                        doc.text("2. LIME Analysis", margin, yOffset);
                        doc.addImage(imgs.lime, 'PNG', margin, yOffset + 5, imgWidth, imgHeight);
                    }

                    // 5. Inject SHAP (Page 3)
                    if (imgs.shap) {
                        doc.addPage(); // Put it on a fresh page
                        yOffset = 20;
                        doc.text("3. SHAP (Integrated Gradients) Analysis", margin, yOffset);
                        doc.addImage(imgs.shap, 'PNG', margin, yOffset + 5, imgWidth, imgHeight);
                    }

                    // 6. Save the file instantly
                    doc.save(`GI_Diagnosis_Report_${Date.now()}.pdf`);

                } catch (err) {
                    console.error("❌ PDF Generation Failed:", err);
                    alert("Error building PDF. Check browser console.");
                } finally {
                    // Instantly restore the button
                    downloadPdfBtn.innerHTML = originalBtnText;
                    downloadPdfBtn.disabled = false;
                }
            }, 100);
        });
    }
});