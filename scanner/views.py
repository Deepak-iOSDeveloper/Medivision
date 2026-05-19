import os
import json
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage

from .models import ScanResult
from .predictor import MODEL_REGISTRY, predict


# ── helpers ───────────────────────────────────────────────────────────────────

def _all_scanners():
    """Return MODEL_REGISTRY as list with key attached, for templates."""
    return [{'key': k, **v} for k, v in MODEL_REGISTRY.items()]


# ── Dashboard ─────────────────────────────────────────────────────────────────

def dashboard(request):
    scanners  = _all_scanners()
    recent    = ScanResult.objects.all()[:10]
    stats = {
        'total':    ScanResult.objects.count(),
        'critical': ScanResult.objects.filter(severity='critical').count(),
        'safe':     ScanResult.objects.filter(severity='safe').count(),
        'high':     ScanResult.objects.filter(severity='high').count(),
    }
    return render(request, 'scanner/dashboard.html', {
        'scanners': scanners,
        'recent':   recent,
        'stats':    stats,
    })


# ── Generic scan page (GET) ───────────────────────────────────────────────────

def scan_page(request, model_key):
    if model_key not in MODEL_REGISTRY:
        return redirect('dashboard')
    info     = MODEL_REGISTRY[model_key]
    scanners = _all_scanners()
    return render(request, 'scanner/scan.html', {
        'model_key': model_key,
        'info':      info,
        'scanners':  scanners,
    })


# ── Run prediction (POST AJAX) ────────────────────────────────────────────────
@csrf_exempt
@require_POST
def run_scan(request, model_key):
    if model_key not in MODEL_REGISTRY:
        return JsonResponse({'error': 'Unknown scanner'}, status=404)

    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image uploaded'}, status=400)

    img_file = request.FILES['image']

    # Validate type
    allowed = {'image/jpeg', 'image/png', 'image/jpg', 'image/webp'}
    if img_file.content_type not in allowed:
        return JsonResponse({'error': 'Only JPEG/PNG images are accepted.'}, status=400)

    # Save to media
    save_path = f'uploads/{model_key}/{img_file.name}'
    saved     = default_storage.save(save_path, img_file)
    full_path = default_storage.path(saved)

    try:
        result = predict(model_key, full_path)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    info = MODEL_REGISTRY[model_key]

    # Persist to DB
    scan = ScanResult.objects.create(
        model_key       = model_key,
        scan_title      = info['title'],
        image           = saved,
        predicted_class = result['predicted_class'],
        confidence      = result['confidence'],
        severity        = result['severity'],
        advice          = result['advice'],
        demo_mode       = result['demo_mode'],
    )

    return JsonResponse({
        'scan_id':        str(scan.id),
        'predicted_class': result['predicted_class'],
        'confidence':     result['confidence'],
        'severity':       result['severity'],
        'advice':         result['advice'],
        'all_scores':     result['all_scores'],
        'demo_mode':      result['demo_mode'],
        'image_url':      scan.image.url,
    })


# ── Result detail page ────────────────────────────────────────────────────────

def result_page(request, scan_id):
    scan     = get_object_or_404(ScanResult, pk=scan_id)
    info     = MODEL_REGISTRY.get(scan.model_key, {})
    scanners = _all_scanners()
    return render(request, 'scanner/result.html', {
        'scan':     scan,
        'info':     info,
        'scanners': scanners,
    })


# ── History page ──────────────────────────────────────────────────────────────

def history(request):
    scans    = ScanResult.objects.all()
    scanners = _all_scanners()
    return render(request, 'scanner/history.html', {
        'scans':    scans,
        'scanners': scanners,
    })


# ── Delete scan ───────────────────────────────────────────────────────────────

@require_POST
def delete_scan(request, scan_id):
    scan = get_object_or_404(ScanResult, pk=scan_id)
    scan.delete()
    return JsonResponse({'ok': True})
